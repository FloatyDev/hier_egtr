# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
import yaml
import os
import sys
import shutil

from glob import glob
from pathlib import Path
from pytorch_lightning.callbacks import Callback

import ipdb
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

from data.open_image import OIDataset, oi_get_statistics
from data.visual_genome import VGDataset, vg_get_statistics
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OIEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from lib.pytorch_misc import argsort_desc
from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentorNoCrop,
    DeformableDetrHungarianMatcher,
)
from model.egtr import DetrForSceneGraphGeneration
from util.box_ops import rescale_bboxes
from util.misc import use_deterministic_algorithms
from model.util import (
    GTTripletVis,
    count_trainable,
    get_super_rel_map,
    get_orig2idx,
    SuperRelationConfusionMatrix,
    ExpertDiagnosticsCallback,
    surgery_initialize_experts,
    run_surgery_sanity_check,
    RouterCalibrationLogger,
)
import wandb

seed_everything(42, workers=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")


def stitch_logits(output_dict, orig2fam_map, device):
    """
    Reconstruct the full (B, N, N, 50) relation-logit tensor.

    Important:
    output_dict["geo"], ["poss"], and ["sem"] already contain
    the corresponding router family logit. Do not add it again.
    """
    logits_super = output_dict["super"]
    batch_size, num_queries, _, _ = logits_super.shape

    num_fine_classes = len(orig2fam_map)

    final_logits = torch.full(
        (
            batch_size,
            num_queries,
            num_queries,
            num_fine_classes,
        ),
        -1000.0,
        dtype=logits_super.dtype,
        device=device,
    )

    geo_indices = [index for index, family in enumerate(orig2fam_map) if family == 0]
    poss_indices = [index for index, family in enumerate(orig2fam_map) if family == 1]
    sem_indices = [index for index, family in enumerate(orig2fam_map) if family == 2]

    if "geo" in output_dict:
        final_logits[..., geo_indices] = output_dict["geo"]

    if "poss" in output_dict:
        final_logits[..., poss_indices] = output_dict["poss"]

    if "sem" in output_dict:
        final_logits[..., sem_indices] = output_dict["sem"]

    return final_logits


def align_predictions_to_ground_truth(
    pred_rel_probs,
    pred_connectivity,
    pred_obj_logits,
    matching_indices,
    num_gt_objects,
    num_obj_labels,
):
    """
    Align DETR query predictions to ground-truth object indexing.

    Args:
        pred_rel_probs:
            Tensor [N_queries, N_queries, N_rel_classes].

        pred_connectivity:
            Tensor [N_queries, N_queries, 1] or
            [N_queries, N_queries].

        pred_obj_logits:
            Tensor [N_queries, N_obj_classes + 1].

        matching_indices:
            Tuple (matched_query_indices, matched_target_indices).

        num_gt_objects:
            Number of GT objects in this image.

        num_obj_labels:
            Number of foreground object classes.

    Returns:
        aligned_rel_probs:
            Tensor [N_gt, N_gt, N_rel_classes].

        aligned_obj_scores:
            Tensor [N_gt].

        aligned_obj_classes:
            Tensor [N_gt].

        gt_to_query:
            Tensor [N_gt], mapping each GT index to a DETR query.
    """
    device = pred_rel_probs.device
    dtype = pred_rel_probs.dtype
    num_rel_classes = pred_rel_probs.shape[-1]

    query_indices, target_indices = matching_indices

    query_indices = query_indices.to(device=device, dtype=torch.long)
    target_indices = target_indices.to(device=device, dtype=torch.long)

    gt_to_query = torch.full(
        (num_gt_objects,),
        -1,
        dtype=torch.long,
        device=device,
    )

    gt_to_query[target_indices] = query_indices

    aligned_rel_probs = torch.zeros(
        (
            num_gt_objects,
            num_gt_objects,
            num_rel_classes,
        ),
        dtype=dtype,
        device=device,
    )

    aligned_obj_scores = torch.zeros(
        num_gt_objects,
        dtype=dtype,
        device=device,
    )

    aligned_obj_classes = torch.zeros(
        num_gt_objects,
        dtype=torch.long,
        device=device,
    )

    valid_gt_indices = torch.where(gt_to_query >= 0)[0]

    if valid_gt_indices.numel() == 0:
        return (
            aligned_rel_probs,
            aligned_obj_scores,
            aligned_obj_classes,
            gt_to_query,
        )

    valid_query_indices = gt_to_query[valid_gt_indices]

    # Align object classification predictions.
    matched_obj_probs = pred_obj_logits[valid_query_indices].softmax(dim=-1)[
        ..., :num_obj_labels
    ]

    valid_obj_scores, valid_obj_classes = matched_obj_probs.max(dim=-1)

    aligned_obj_scores[valid_gt_indices] = valid_obj_scores
    aligned_obj_classes[valid_gt_indices] = valid_obj_classes

    # Align relation predictions through indexed extraction.
    subject_queries = valid_query_indices[:, None].expand(
        -1,
        valid_query_indices.numel(),
    )
    object_queries = valid_query_indices[None, :].expand(
        valid_query_indices.numel(),
        -1,
    )

    aligned_valid_rel_probs = pred_rel_probs[
        subject_queries,
        object_queries,
    ]

    # Apply the model's learned connectivity gate.
    if pred_connectivity is not None:
        connectivity = pred_connectivity

        if connectivity.dim() == 3 and connectivity.shape[-1] == 1:
            connectivity = connectivity.squeeze(-1)

        aligned_connectivity = connectivity[
            subject_queries,
            object_queries,
        ]

        aligned_valid_rel_probs = (
            aligned_valid_rel_probs * aligned_connectivity.unsqueeze(-1)
        )

    aligned_rel_probs[
        valid_gt_indices[:, None],
        valid_gt_indices[None, :],
    ] = aligned_valid_rel_probs

    # Never evaluate self-relations.
    diagonal = torch.arange(num_gt_objects, device=device)
    aligned_rel_probs[diagonal, diagonal] = 0.0

    return (
        aligned_rel_probs,
        aligned_obj_scores,
        aligned_obj_classes,
        gt_to_query,
    )


def select_top_relation_pairs(
    relation_probs,
    object_scores=None,
    max_topk=100,
):
    """
    Rank ordered subject-object pairs.

    Args:
        relation_probs:
            Tensor [N, N, num_rel_classes].

        object_scores:
            Optional tensor [N]. When supplied, pair scores include
            subject and object classification confidence.

        max_topk:
            Maximum number of ordered pairs returned.

    Returns:
        pred_rel_inds:
            NumPy array [K, 2].

        rel_scores:
            NumPy array [K, num_rel_classes].
    """
    num_objects = relation_probs.shape[0]

    if num_objects <= 1:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(
                (0, relation_probs.shape[-1]),
                dtype=np.float32,
            ),
        )

    relation_max_scores = relation_probs.max(dim=-1).values

    if object_scores is not None:
        pair_object_scores = torch.outer(
            object_scores,
            object_scores,
        )
        triplet_scores = pair_object_scores * relation_max_scores
    else:
        triplet_scores = relation_max_scores

    diagonal = torch.arange(
        num_objects,
        device=triplet_scores.device,
    )
    triplet_scores[diagonal, diagonal] = 0.0

    pred_rel_inds = argsort_desc(triplet_scores.detach().cpu().numpy())[:max_topk, :]

    rel_scores = relation_probs[
        pred_rel_inds[:, 0],
        pred_rel_inds[:, 1],
    ]

    return (
        pred_rel_inds,
        rel_scores.detach().cpu().numpy(),
    )


def evaluate_entry(
    mode,
    gt_entry,
    pred_entry,
    single_sgg_evaluator,
    single_sgg_evaluator_list,
):
    single_sgg_evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    for pred_id, _, evaluator_rel in single_sgg_evaluator_list:
        predicate_mask = np.in1d(
            gt_entry["gt_relations"][:, -1],
            pred_id,
        )

        filtered_relations = gt_entry["gt_relations"][predicate_mask]

        if filtered_relations.shape[0] == 0:
            continue

        gt_entry_rel = {
            "gt_relations": filtered_relations,
            "gt_boxes": gt_entry["gt_boxes"],
            "gt_classes": gt_entry["gt_classes"],
        }

        evaluator_rel[mode].evaluate_scene_graph_entry(
            gt_entry_rel,
            pred_entry,
        )


def evaluate_batch_all_modes(
    outputs,
    targets,
    matcher,
    eval_modes,
    num_obj_labels,
    single_sgg_evaluator,
    single_sgg_evaluator_list,
    max_topk=100,
):
    """
    Evaluate a model batch under SGDet, SGCls, and PredCls.
    """
    valid_modes = {"sgdet", "sgcls", "predcls"}

    for mode in eval_modes:
        if mode not in valid_modes:
            raise ValueError(f"Unsupported evaluation mode: {mode}")

    pred_rel_raw = outputs["pred_rel"]

    # Hierarchical output is a dictionary of expert tensors.
    if isinstance(pred_rel_raw, dict):
        from model.util import get_super_rel_map

        pred_rel_logits = stitch_logits(
            pred_rel_raw,
            get_super_rel_map(),
            outputs["logits"].device,
        )
    else:
        pred_rel_logits = pred_rel_raw

    # The matcher only needs object logits and predicted boxes.
    matcher_outputs = {
        "logits": outputs["logits"],
        "pred_boxes": outputs["pred_boxes"],
    }

    # Targets must be on the same device as matcher outputs.
    matcher_targets = []

    for target in targets:
        matcher_target = {}

        for key, value in target.items():
            if torch.is_tensor(value):
                matcher_target[key] = value.to(outputs["logits"].device)
            else:
                matcher_target[key] = value

        matcher_targets.append(matcher_target)

    match_indices = matcher(
        matcher_outputs,
        matcher_targets,
    )

    for batch_index, target in enumerate(targets):
        gt_entry = build_gt_entry(target)

        if gt_entry["gt_relations"].shape[0] == 0:
            continue

        orig_size = target["orig_size"].cpu()
        num_gt_objects = len(gt_entry["gt_classes"])

        pred_obj_logits = outputs["logits"][batch_index]
        pred_boxes_normalized = outputs["pred_boxes"][batch_index]

        # Hierarchical predictions are logits.
        # Flat EGTR already returns activated predicate scores.
        if isinstance(pred_rel_raw, dict):
            pred_rel_probs = pred_rel_logits[batch_index].softmax(dim=-1)
        else:
            pred_rel_probs = pred_rel_logits[batch_index]

        pred_connectivity = None

        if outputs.get("pred_connectivity") is not None:
            pred_connectivity = torch.clamp(
                outputs["pred_connectivity"][batch_index],
                min=0.0,
                max=1.0,
            )

        (
            aligned_rel_probs,
            aligned_obj_scores,
            aligned_obj_classes,
            gt_to_query,
        ) = align_queries_to_gt(
            pred_rel_probs=pred_rel_probs,
            pred_connectivity=pred_connectivity,
            pred_obj_logits=pred_obj_logits,
            match_indices=match_indices[batch_index],
            num_gt_objects=num_gt_objects,
            num_obj_labels=num_obj_labels,
        )

        num_unmatched = int((gt_to_query < 0).sum().item())

        if num_unmatched > 0:
            print(
                "[evaluation] Warning: "
                f"{num_unmatched}/{num_gt_objects} "
                "GT objects were not matched."
            )

        for mode in eval_modes:
            if mode == "sgdet":
                pred_obj_probs = pred_obj_logits.softmax(dim=-1)[..., :num_obj_labels]

                obj_scores, pred_classes = pred_obj_probs.max(dim=-1)

                sgdet_rel_probs = pred_rel_probs

                if pred_connectivity is not None:
                    connectivity = pred_connectivity

                    if connectivity.dim() == 3 and connectivity.shape[-1] == 1:
                        connectivity = connectivity.squeeze(-1)

                    sgdet_rel_probs = sgdet_rel_probs * connectivity.unsqueeze(-1)

                pred_rel_inds, rel_scores = select_top_pairs(
                    rel_probs=sgdet_rel_probs,
                    obj_scores=obj_scores,
                    max_topk=max_topk,
                )

                pred_boxes = (
                    rescale_bboxes(
                        pred_boxes_normalized.cpu(),
                        torch.flip(orig_size, dims=[0]),
                    )
                    .clone()
                    .numpy()
                )

                pred_entry = {
                    "pred_boxes": pred_boxes,
                    "pred_classes": (pred_classes.detach().cpu().numpy()),
                    "obj_scores": (obj_scores.detach().cpu().numpy()),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            elif mode == "sgcls":
                pred_rel_inds, rel_scores = select_top_pairs(
                    rel_probs=aligned_rel_probs,
                    obj_scores=aligned_obj_scores,
                    max_topk=max_topk,
                )

                pred_entry = {
                    # Evaluator uses GT boxes in SGCls.
                    "pred_boxes": gt_entry["gt_boxes"],
                    "pred_classes": (aligned_obj_classes.detach().cpu().numpy()),
                    "obj_scores": (aligned_obj_scores.detach().cpu().numpy()),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            elif mode == "predcls":
                pred_rel_inds, rel_scores = select_top_pairs(
                    rel_probs=aligned_rel_probs,
                    obj_scores=None,
                    max_topk=max_topk,
                )

                pred_entry = {
                    # Evaluator uses GT boxes and GT classes.
                    "pred_boxes": gt_entry["gt_boxes"],
                    "pred_classes": gt_entry["gt_classes"],
                    "obj_scores": np.ones(
                        num_gt_objects,
                        dtype=np.float32,
                    ),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            else:
                raise RuntimeError(f"Unhandled evaluation mode: {mode}")

            if mode in {"sgcls", "predcls"}:
                if pred_rel_inds.size > 0:
                    assert pred_rel_inds.max() < num_gt_objects, (
                        f"{mode} relation indices are not " "GT-object indices."
                    )

            assert (
                pred_entry["pred_rel_inds"].shape[0]
                == pred_entry["rel_scores"].shape[0]
            )

            evaluate_entry(
                mode=mode,
                gt_entry=gt_entry,
                pred_entry=pred_entry,
                single_sgg_evaluator=single_sgg_evaluator,
                single_sgg_evaluator_list=(single_sgg_evaluator_list),
            )


def evaluate_batch(
    outputs,
    targets,
    matching_indices,
    eval_modes,
    multiple_sgg_evaluator=None,
    multiple_sgg_evaluator_list=None,
    single_sgg_evaluator=None,
    single_sgg_evaluator_list=None,
    oi_evaluator=None,
    num_obj_labels=None,
    max_topk=100,
    **_,
):
    """
    Evaluate one batch under SGDet, SGCls, and/or PredCls.

    Protocols:
        SGDet:
            Predicted boxes, object classes, and predicates.

        SGCls:
            GT boxes, predicted object classes, predicted predicates.

        PredCls:
            GT boxes, GT object classes, predicted predicates.
    """
    if num_obj_labels is None:
        num_obj_labels = outputs["logits"].shape[-1] - 1

    valid_modes = {"sgdet", "sgcls", "predcls"}

    for mode in eval_modes:
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid evaluation mode: {mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

    orig2fam = get_super_rel_map()
    pred_rel_raw = outputs["pred_rel"]

    if isinstance(pred_rel_raw, dict):
        pred_rel_logits = stitch_logits(
            pred_rel_raw,
            orig2fam,
            outputs["logits"].device,
        )
    else:
        pred_rel_logits = pred_rel_raw

    for batch_index, target in enumerate(targets):
        pred_obj_logits = outputs["logits"][batch_index]
        pred_boxes_normalized = outputs["pred_boxes"][batch_index]

        # Hierarchical heads produce logits, so use softmax.
        # For flat EGTR, preserve the existing tensor behavior.
        if isinstance(pred_rel_raw, dict):
            pred_rel_probs = pred_rel_logits[batch_index].softmax(dim=-1)
        else:
            pred_rel_probs = pred_rel_logits[batch_index]

        pred_connectivity = None
        if "pred_connectivity" in outputs:
            pred_connectivity = torch.clamp(
                outputs["pred_connectivity"][batch_index],
                min=0.0,
                max=1.0,
            )

        orig_size = target["orig_size"].cpu()
        target_labels = target["class_labels"].cpu()
        target_boxes = target["boxes"].cpu()
        target_rel = target["rel"].cpu().nonzero()

        gt_boxes = (
            rescale_bboxes(
                target_boxes,
                torch.flip(orig_size, dims=[0]),
            )
            .clone()
            .numpy()
        )

        gt_classes = target_labels.clone().numpy()

        gt_entry = {
            "gt_relations": target_rel.clone().numpy(),
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
        }

        # Some images may contain no annotated relations.
        if gt_entry["gt_relations"].shape[0] == 0:
            continue

        num_gt_objects = len(gt_classes)

        # Build GT-aligned query predictions once per image.
        (
            aligned_rel_probs,
            aligned_obj_scores,
            aligned_obj_classes,
            gt_to_query,
        ) = align_predictions_to_ground_truth(
            pred_rel_probs=pred_rel_probs,
            pred_connectivity=pred_connectivity,
            pred_obj_logits=pred_obj_logits,
            matching_indices=matching_indices[batch_index],
            num_gt_objects=num_gt_objects,
            num_obj_labels=num_obj_labels,
        )

        num_unmatched = int((gt_to_query < 0).sum().item())

        if num_unmatched > 0:
            print(
                f"[evaluation] Warning: image has "
                f"{num_unmatched}/{num_gt_objects} unmatched GT objects."
            )

        for mode in eval_modes:
            if mode == "sgdet":
                pred_obj_probs = pred_obj_logits.softmax(dim=-1)[..., :num_obj_labels]

                obj_scores, pred_classes = pred_obj_probs.max(dim=-1)

                sgdet_rel_probs = pred_rel_probs

                if pred_connectivity is not None:
                    connectivity = pred_connectivity

                    if connectivity.dim() == 3 and connectivity.shape[-1] == 1:
                        connectivity = connectivity.squeeze(-1)

                    sgdet_rel_probs = sgdet_rel_probs * connectivity.unsqueeze(-1)

                pred_rel_inds, rel_scores = select_top_relation_pairs(
                    relation_probs=sgdet_rel_probs,
                    object_scores=obj_scores,
                    max_topk=max_topk,
                )

                pred_boxes = (
                    rescale_bboxes(
                        pred_boxes_normalized.cpu(),
                        torch.flip(orig_size, dims=[0]),
                    )
                    .clone()
                    .numpy()
                )

                pred_entry = {
                    "pred_boxes": pred_boxes,
                    "pred_classes": (pred_classes.detach().cpu().numpy()),
                    "obj_scores": (obj_scores.detach().cpu().numpy()),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            elif mode == "sgcls":
                pred_rel_inds, rel_scores = select_top_relation_pairs(
                    relation_probs=aligned_rel_probs,
                    object_scores=aligned_obj_scores,
                    max_topk=max_topk,
                )

                pred_entry = {
                    # The evaluator replaces these with GT boxes.
                    "pred_boxes": gt_boxes,
                    "pred_classes": (aligned_obj_classes.detach().cpu().numpy()),
                    "obj_scores": (aligned_obj_scores.detach().cpu().numpy()),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            elif mode == "predcls":
                # In PredCls, object categories are known and should
                # not influence pair ranking through confidence.
                pred_rel_inds, rel_scores = select_top_relation_pairs(
                    relation_probs=aligned_rel_probs,
                    object_scores=None,
                    max_topk=max_topk,
                )

                pred_entry = {
                    "pred_boxes": gt_boxes,
                    "pred_classes": gt_classes,
                    "obj_scores": np.ones(
                        num_gt_objects,
                        dtype=np.float32,
                    ),
                    "pred_rel_inds": pred_rel_inds,
                    "rel_scores": rel_scores,
                }

            else:
                raise RuntimeError(f"Unhandled evaluation mode: {mode}")

            # Overall unconstrained/multiple-predicate evaluator.
            if multiple_sgg_evaluator is not None:
                multiple_sgg_evaluator[mode].evaluate_scene_graph_entry(
                    gt_entry,
                    pred_entry,
                )

            # Per-predicate unconstrained evaluator.
            if multiple_sgg_evaluator_list is not None:
                for (
                    pred_id,
                    _,
                    evaluator_rel,
                ) in multiple_sgg_evaluator_list:
                    gt_entry_rel = {
                        "gt_relations": gt_entry["gt_relations"].copy(),
                        "gt_boxes": gt_entry["gt_boxes"],
                        "gt_classes": gt_entry["gt_classes"],
                    }

                    mask = np.in1d(
                        gt_entry_rel["gt_relations"][:, -1],
                        pred_id,
                    )

                    gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask]

                    if gt_entry_rel["gt_relations"].shape[0] == 0:
                        continue

                    evaluator_rel[mode].evaluate_scene_graph_entry(
                        gt_entry_rel,
                        pred_entry,
                    )

            # Overall graph-constrained evaluator.
            if single_sgg_evaluator is not None:
                single_sgg_evaluator[mode].evaluate_scene_graph_entry(
                    gt_entry,
                    pred_entry,
                )

            # Per-predicate graph-constrained evaluator.
            if single_sgg_evaluator_list is not None:
                for (
                    pred_id,
                    _,
                    evaluator_rel,
                ) in single_sgg_evaluator_list:
                    gt_entry_rel = {
                        "gt_relations": gt_entry["gt_relations"].copy(),
                        "gt_boxes": gt_entry["gt_boxes"],
                        "gt_classes": gt_entry["gt_classes"],
                    }

                    mask = np.in1d(
                        gt_entry_rel["gt_relations"][:, -1],
                        pred_id,
                    )

                    gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask]

                    if gt_entry_rel["gt_relations"].shape[0] == 0:
                        continue

                    evaluator_rel[mode].evaluate_scene_graph_entry(
                        gt_entry_rel,
                        pred_entry,
                    )

            # Open Images evaluation remains SGDet-only.
            if oi_evaluator is not None and mode == "sgdet":
                oi_scores = rel_scores.max(axis=1)

                pred_entry_oi = {
                    **pred_entry,
                    "sbj_obj_inds": pred_rel_inds,
                    "pred_scores": oi_scores,
                }

                oi_evaluator(gt_entry, pred_entry_oi)


## Reference: https://github.com/yuweihao/KERN/blob/master/models/eval_rels.py
# def evaluate_batch(
#    outputs,
#    targets,
#    sgg_evaluator,
#    sgg_evaluator_list,
#    num_obj_labels,
#    max_topk=100,
# ):
#    orig2fam = get_super_rel_map()
#    pred_rel_raw = outputs["pred_rel"]
#
#    if isinstance(pred_rel_raw, dict):
#        # We are in Hierarchical Mode: Stitch to get (B, N, N, 50)
#        pred_rel_logits = stitch_logits(
#            pred_rel_raw, orig2fam, outputs["logits"].device
#        )
#    else:
#        pred_rel_logits = pred_rel_raw
#
#    for j, target in enumerate(targets):
#        pred_obj_logits = outputs["logits"][j]
#        pred_boxes = outputs["pred_boxes"][j]
#
#        pred_rel_probs = pred_rel_logits[j].softmax(-1)
#
#        orig_size = target["orig_size"].cpu()
#
#        obj_scores, pred_classes = torch.max(
#            pred_obj_logits.softmax(-1)[:, :num_obj_labels], -1
#        )
#
#        sub_ob_scores = torch.outer(obj_scores, obj_scores)
#        sub_ob_scores.fill_diagonal_(0.0)
#
#        if "pred_connectivity" in outputs:
#            pred_connectivity = torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0)
#            pred_rel_probs = pred_rel_probs * pred_connectivity
#
#        rel_max_scores, _ = pred_rel_probs.max(dim=-1)
#        triplet_scores = sub_ob_scores * rel_max_scores
#
#        pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[:max_topk, :]
#
#        rel_scores = (
#            pred_rel_probs.cpu()
#            .clone()
#            .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]]
#        )
#
#        pred_entry = {
#            "pred_boxes": rescale_bboxes(
#                pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
#            )
#            .clone()
#            .numpy(),
#            "pred_classes": pred_classes.cpu().clone().numpy(),
#            "obj_scores": obj_scores.cpu().clone().numpy(),
#            "pred_rel_inds": pred_rel_inds,
#            "rel_scores": rel_scores,
#        }
#
#        target_labels = target["class_labels"].cpu()
#        target_boxes = target["boxes"].cpu()
#        target_rel = target["rel"].cpu().nonzero()
#
#        gt_entry = {
#            "gt_relations": target_rel.clone().numpy(),
#            "gt_boxes": rescale_bboxes(target_boxes, torch.flip(orig_size, dims=[0]))
#            .clone()
#            .numpy(),
#            "gt_classes": target_labels.clone().numpy(),
#        }
#
#        if sgg_evaluator is not None:
#            sgg_evaluator["sgdet"].evaluate_scene_graph_entry(gt_entry, pred_entry)
#
#        if sgg_evaluator_list is not None:
#            for pred_id, _, evaluator_rel in sgg_evaluator_list:
#                gt_entry_rel = gt_entry.copy()
#                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
#                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
#                if gt_entry_rel["gt_relations"].shape[0] == 0:
#                    continue
#                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
#                    gt_entry_rel, pred_entry
#                )


def collate_fn(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


class SGG(pl.LightningModule):
    def __init__(
        self,
        architecture,
        backbone_dirpath,
        auxiliary_loss,
        lr,
        lr_backbone,
        lr_initialized,
        weight_decay,
        pretrained,
        main_trained,
        from_scratch,
        id2label,
        rel_loss_coefficient,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        rel_categories,
        multiple_sgg_evaluator,
        multiple_sgg_evaluator_list,
        single_sgg_evaluator,
        single_sgg_evaluator_list,
        coco_evaluator,
        oi_evaluator,
        feature_extractor,
        num_queries,
        ce_loss_coefficient,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
        use_freq_bias,
        fg_matrix,
        use_log_softmax,
        freq_bias_eps,
        connectivity_loss_coefficient,
        logit_adjustment,
        logit_adj_tau,
        hierarchical,
        num_geometric,
        num_possessive,
        num_semantic,
        num_negatives,
        super_weight,
        train_relation_head=False,
        artifact_path="",
        flat_path="",
        use_class_context=False,
    ):

        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(pretrained)
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.from_scratch = from_scratch
        config.num_rel_labels = len(rel_categories)
        config.num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.rel_loss_coefficient = rel_loss_coefficient
        config.smoothing = smoothing
        config.rel_sample_negatives = rel_sample_negatives
        config.rel_sample_nonmatching = rel_sample_nonmatching
        config.ce_loss_coefficient = ce_loss_coefficient
        config.pretrained = pretrained
        config.rel_sample_negatives_largest = rel_sample_negatives_largest
        config.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest

        config.connectivity_loss_coefficient = connectivity_loss_coefficient
        config.use_freq_bias = use_freq_bias
        config.use_log_softmax = use_log_softmax
        config.freq_bias_eps = freq_bias_eps

        config.logit_adjustment = logit_adjustment
        config.logit_adj_tau = logit_adj_tau

        # Hierarchical plug-in additions
        config.hierarchical = hierarchical
        config.num_geometric = num_geometric
        config.num_possessive = num_possessive
        config.num_semantic = num_semantic
        config.num_negatives = num_negatives
        config.super_weight = super_weight
        config.use_class_context = use_class_context
        config.flat_path = flat_path
        config.id2label = id2label

        self.config = config

        if config.from_scratch:
            assert backbone_dirpath
            self.model = DetrForSceneGraphGeneration(config=config, fg_matrix=fg_matrix)
            self.model.model.backbone.load_state_dict(
                torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
            )
            self.initialized_keys = []
        else:
            self.model, load_info = DetrForSceneGraphGeneration.from_pretrained(
                pretrained,
                config=config,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                fg_matrix=fg_matrix,
            )
            self.initialized_keys = load_info["missing_keys"] + [
                _key for _key, _, _ in load_info["mismatched_keys"]
            ]

        if train_relation_head:
            if not main_trained:
                # This restores the 'shared_layers' and 'super_head' learned during distillation
                assert (
                    artifact_path
                ), "Must provide artifact_path to the Super-Classifier checkpoint"
                print(f"[sgg] Loading Router weights from: {artifact_path}")

                ckpt_path = sorted(
                    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
                    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
                )[-1]
                print(f"Found Router checkpoint: {ckpt_path}")

                state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict

                # strict=False is expected because 'expert_geo', 'expert_poss', etc. are missing
                missing, unexpected = self.model.load_state_dict(
                    state_dict, strict=False
                )

                print("\n[sgg] Router Loading Report:")
                print(
                    f"   - Missing Keys (Should be Experts): {[k for k in missing if 'expert' in k]}"
                )
                print(
                    f"   - Unexpected Keys (Should be Fine Head): {[k for k in unexpected if 'fine_head' in k]}"
                )

                if any("super_head" in m for m in missing):
                    print(
                        "WARNING: 'super_head' missing! It will be random (Distillation lost)."
                    )
                if any("shared_layers" in m for m in missing):
                    print(
                        "CRITICAL WARNING: 'shared_layers' missing! Feature extractor is random."
                    )

                if flat_path:
                    print(
                        f"\n[sgg] Performing Weight Surgery from Flat-50: {flat_path}"
                    )

                    flat_ckpt = sorted(
                        glob(f"{flat_path}/checkpoints/epoch=*.ckpt"),
                        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
                    )[-1]
                    print(f"Found Flat checkpoint: {flat_ckpt}")

                    # Perform Surgery: Slice weights -> Inject into Experts
                    surgery_initialize_experts(self.model, flat_ckpt, device="cpu")

                    # Verify: Ensure weights match exactly
                    run_surgery_sanity_check(self.model, flat_ckpt, device="cpu")
                else:
                    print(
                        "\n[sgg] WARNING: No 'flat_path' provided. Experts will be RANDOM INITIALIZED."
                    )

            print("\n[sgg] Configuring Gradients...")

            # Freeze everything initially
            for p in self.model.parameters():
                p.requires_grad = False

            # Define which heads are trainable
            trainable_modules = [
                "rel_predictor.super_head",
                "rel_predictor.expert_geo",
                "rel_predictor.expert_poss",
                "rel_predictor.expert_sem",
                "rel_predictor.shared_layers",
            ]

            # Unfreeze specific modules
            for n, p in self.model.named_parameters():
                if any(t in n for t in trainable_modules):
                    p.requires_grad = True

            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]

            # Validation assertions
            assert any(
                "expert_geo" in t for t in trainable
            ), "Experts are not trainable!"
            assert any("super_head" in t for t in trainable), "Super Head is frozen!"
            # We explicitly want shared_layers FROZEN to prevent 'shock' from the new experts destroying the router features
            # assert not any(
            #    "shared_layers" in t for t in trainable
            # ), "Shared Layers leaked into training!"

            count_trainable(model=self.model, debugging=True)
            print(f"[sgg] Final Trainable Parameter Count: {len(trainable)}")

        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_initialized = lr_initialized
        self.weight_decay = weight_decay
        self.multiple_sgg_evaluator = multiple_sgg_evaluator
        self.multiple_sgg_evaluator_list = multiple_sgg_evaluator_list
        self.single_sgg_evaluator = single_sgg_evaluator
        self.single_sgg_evaluator_list = single_sgg_evaluator_list
        self.coco_evaluator = coco_evaluator
        self.oi_evaluator = oi_evaluator
        self.feature_extractor = feature_extractor

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        # 1. Run the shared forward pass
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)

        if batch_idx % 50 == 0 and loss.requires_grad:

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            if trainable_params:
                if (
                    "loss_rel_ce" in loss_dict
                    and loss_dict["loss_rel_ce"].requires_grad
                ):
                    grads_ce = torch.autograd.grad(
                        loss_dict["loss_rel_ce"],
                        trainable_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    norm_ce = torch.norm(
                        torch.stack(
                            [
                                torch.norm(g.detach(), 2)
                                for g in grads_ce
                                if g is not None
                            ]
                        )
                    )
                    self.log(
                        "grads/debug_norm_rel_ce",
                        norm_ce,
                        prog_bar=False,
                        sync_dist=True,
                    )

                if (
                    "loss_rel_distill" in loss_dict
                    and loss_dict["loss_rel_distill"].requires_grad
                ):
                    grads_distill = torch.autograd.grad(
                        loss_dict["loss_rel_distill"],
                        trainable_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    norm_distill = torch.norm(
                        torch.stack(
                            [
                                torch.norm(g.detach(), 2)
                                for g in grads_distill
                                if g is not None
                            ]
                        )
                    )
                    self.log(
                        "grads/debug_norm_rel_distill",
                        norm_distill,
                        prog_bar=False,
                        sync_dist=True,
                    )

                    grads_total = torch.autograd.grad(
                        loss,
                        trainable_params,
                        retain_graph=True,  # Must retain for the actual optimizer.step()!
                        allow_unused=True,
                    )
                    norm_total = torch.norm(
                        torch.stack(
                            [
                                torch.norm(g.detach(), 2)
                                for g in grads_total
                                if g is not None
                            ]
                        )
                    )
                    self.log(
                        "grads/debug_norm_total",
                        norm_total,
                        prog_bar=True,
                        sync_dist=True,
                    )

        # Log standard metrics
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"training_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2 norm for each layer
        grad_norms = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None and "rel_predictor" in name:
                grad_norms[name] = p.grad.norm(2).item()

        self.log_dict({f"grads/{k}": v for k, v in grad_norms.items()})

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        return {"outputs": outputs, "targets": batch["labels"]}

    @rank_zero_only
    def on_train_start(self) -> None:
        if hasattr(self, "logger") and self.logger is not None:
            log_dir = self.logger.log_dir
        else:
            log_dir = os.getcwd()

        self.config.save_pretrained(log_dir)
        return super().on_train_start()

    def test_step(self, batch, batch_idx):

        self.model.eval()

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        targets = [{k: v.cpu() for k, v in label.items()} for label in batch["labels"]]

        with torch.no_grad():
            outputs = self.forward(pixel_values, pixel_mask)
            # eval SGG
            evaluate_batch(
                outputs,
                targets,
                self.multiple_sgg_evaluator,
                self.multiple_sgg_evaluator_list,
                self.single_sgg_evaluator,
                self.single_sgg_evaluator_list,
                self.oi_evaluator,
                self.config.num_labels,
                hierarchical=self.config.hierarchical,
                orig2fam=get_super_rel_map(),
                orig2famidx=get_orig2idx()[0],
            )
            # eval OD
            if self.coco_evaluator is not None:
                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = self.feature_extractor.post_process(
                    outputs, orig_target_sizes.to(self.device)
                )
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                self.coco_evaluator.update(res)

    def on_test_epoch_end(self):
        log_dict = {}

        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            log_dict.update({"AP50": self.coco_evaluator.coco_eval["bbox"].stats[1]})

        # log SGG
        if self.multiple_sgg_evaluator is not None:
            recall = self.multiple_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.single_sgg_evaluator is not None:
            recall = self.single_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.single_sgg_evaluator_list, "sgdet", multiple_preds=False
            )
            recall = dict(zip(["(single)" + x for x in recall.keys()], recall.values()))
            mean_recall = dict(
                zip(["(single)" + x for x in mean_recall.keys()], mean_recall.values())
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.oi_evaluator is not None:
            metrics = self.oi_evaluator.aggregate_metrics()
            log_dict.update(metrics)

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]

        if self.lr_initialized is not None:  # rel_predictor
            initialized_lr_params = self.initialized_keys
        else:
            initialized_lr_params = []
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params))
                    and (not any(nd in n for nd in initialized_lr_params))
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        if initialized_lr_params:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in initialized_lr_params)
                        and p.requires_grad
                    ],
                    "lr": self.lr_initialized,
                }
            )
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser(parser):
    # Your existing args
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--backbone_dirpath", type=str, default="")
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=False)
    parser.add_argument("--from_scratch", type=str2bool, default=False)
    parser.add_argument("--pretrained", type=str, required=False)

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)
    parser.add_argument("--rel_loss_coefficient", type=float, default=15.0)
    parser.add_argument("--connectivity_loss_coefficient", type=float, default=30.0)
    parser.add_argument("--smoothing", type=float, default=1e-14)
    parser.add_argument("--rel_sample_negatives", type=int, default=80)
    parser.add_argument("--rel_sample_nonmatching", type=int, default=80)
    parser.add_argument("--rel_sample_negatives_largest", type=str2bool, default=True)
    parser.add_argument("--rel_sample_nonmatching_largest", type=str2bool, default=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_epochs_finetune", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-7)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr_initialized", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)
    parser.add_argument("--filter_duplicate_rels", type=str2bool, default=True)
    parser.add_argument("--filter_multiple_rels", type=str2bool, default=True)
    parser.add_argument("--use_freq_bias", type=str2bool, default=True)
    parser.add_argument("--use_log_softmax", type=str2bool, default=False)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=["sgdet"],
        choices=["sgdet", "sgcls", "predcls"],
        help=(
            "Scene graph protocols to evaluate. "
            "Example: --eval_modes sgdet sgcls predcls"
        ),
    )
    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--hierarchical", type=str2bool, default=False)
    parser.add_argument("--num_geometric", type=int, default=15)
    parser.add_argument("--num_possessive", type=int, default=11)
    parser.add_argument("--num_semantic", type=int, default=24)
    parser.add_argument("--num_negatives", type=int, default=49)
    parser.add_argument("--super_weight", type=int, default=1)
    parser.add_argument("--train_head", type=str2bool, default=False)
    parser.add_argument("--artifact_path", type=str, default="")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--use_class_context", type=str2bool, default=False)
    parser.add_argument("--flat_path", type=str, default="artifacts/")

    return parser


def parse_args():
    # 1) Parse only --config first (avoid required errors)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = build_parser(config_parser)

    if config_args.config:
        with open(config_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # Optional: warn on unknown keys
        valid_keys = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid_keys
        if unknown:
            print(
                f"Warning: unknown config keys ignored: {sorted(unknown)}",
                file=sys.stderr,
            )

        parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid_keys})

    # CLI overrides config.yaml
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    if args.from_scratch:
        args.pretrained = args.architecture

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )

    # Dataset
    if "visual_genome" in args.data_path:
        train_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="val",
            num_object_queries=args.num_queries,
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 149
        fg_matrix = vg_get_statistics(train_dataset, must_overlap=True)
        print(id2label)
        assert 0
    else:
        train_dataset = OIDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            filter_duplicate_rels=args.filter_duplicate_rels,
            filter_multiple_rels=args.filter_multiple_rels,
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = OIDataset(
            data_folder=args.data_path,
            split="val",
            num_object_queries=args.num_queries,
            feature_extractor=feature_extractor,
        )
        id2label = train_dataset.classes_to_ind  # 0 ~ 600
        fg_matrix = oi_get_statistics(train_dataset, must_overlap=True)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    rel_categories = train_dataset.rel_categories
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    coco_evaluator = None
    oi_evaluator = None

    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if args.eval_when_train_end:
        if args.eval_multiple_preds:
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=True
            )  # R@k
            for index, name in enumerate(rel_categories):
                multiple_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=True),
                    )
                )
        if args.eval_single_preds:
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=False
            )  # R@k
            for index, name in enumerate(rel_categories):
                single_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=False),
                    )
                )
        if "visual_genome" in args.data_path:
            coco_evaluator = CocoEvaluator(
                val_dataset.coco, ["bbox"]
            )  # initialize evaluator with ground truths
        elif "open-image" in args.data_path:
            oi_evaluator = OIEvaluator(
                train_dataset.rel_categories, train_dataset.ind_to_classes
            )

    # Logger setting
    save_dir = f"{args.output_path}/egtr__{'/'.join(args.pretrained.split('/')[-3:]).replace('/', '__')}"
    if args.from_scratch:
        save_dir += "__from_scratch"
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr_bbone__{args.lr_backbone}__lr{args.lr}__lr_init{args.lr_initialized}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.hierarchical:
        name += "__hier"
    if args.train_head:
        name += "__train_rel_head"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    tensorboard_logger = TensorBoardLogger(save_dir, name=name, version=version)

    # initialize wandblogger
    wandb_logger = WandbLogger(
        project="hier-egtr_distill_training",
        log_model=False,
        save_dir="./logs",
        name=name,
    )
    logger_list = [tensorboard_logger, wandb_logger]
    if os.path.exists(f"{tensorboard_logger.log_dir}/checkpoints"):
        if os.path.exists(f"{tensorboard_logger.log_dir}/checkpoints/last.ckpt"):
            ckpt_path = f"{tensorboard_logger.log_dir}/checkpoints/last.ckpt"
        else:
            ckpt_path = sorted(
                glob(f"{tensorboard_logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]
    else:
        ckpt_path = None
    print(f"ckpt_path for resume: {ckpt_path}")
    # Module
    module = SGG(
        architecture=args.architecture,
        backbone_dirpath=args.backbone_dirpath,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        lr_initialized=args.lr_initialized,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        main_trained="",
        from_scratch=args.from_scratch,
        id2label=id2label,
        rel_loss_coefficient=args.rel_loss_coefficient,
        smoothing=args.smoothing,
        rel_sample_negatives=args.rel_sample_negatives,
        rel_sample_nonmatching=args.rel_sample_nonmatching,
        rel_categories=rel_categories,
        multiple_sgg_evaluator=multiple_sgg_evaluator,
        multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
        single_sgg_evaluator=single_sgg_evaluator,
        single_sgg_evaluator_list=single_sgg_evaluator_list,
        coco_evaluator=coco_evaluator,
        oi_evaluator=oi_evaluator,
        feature_extractor=feature_extractor,
        num_queries=args.num_queries,
        ce_loss_coefficient=args.ce_loss_coefficient,
        rel_sample_negatives_largest=args.rel_sample_negatives_largest,
        rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
        use_freq_bias=args.use_freq_bias,
        fg_matrix=fg_matrix,
        use_log_softmax=args.use_log_softmax,
        freq_bias_eps=1e-12,
        connectivity_loss_coefficient=args.connectivity_loss_coefficient,
        logit_adjustment=args.logit_adjustment,
        logit_adj_tau=args.logit_adj_tau,
        hierarchical=args.hierarchical,
        num_geometric=args.num_geometric,
        num_possessive=args.num_possessive,
        num_semantic=args.num_semantic,
        num_negatives=args.num_negatives,
        super_weight=args.super_weight,
        train_relation_head=args.train_head,
        artifact_path=args.artifact_path,
        flat_path=args.flat_path,
        use_class_context=args.use_class_context,
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    visualize_imgs = GTTripletVis(
        dataset=train_dataset,
        id2label=id2label,
        rel_categories=rel_categories,
        freq=1,
    )
    cm_callback = SuperRelationConfusionMatrix(id2label=id2label)
    expert_callback = ExpertDiagnosticsCallback()
    router_calibration_cb = RouterCalibrationLogger(
        device="cuda", rel_categories=rel_categories
    )

    class SaveConfigCallback(Callback):
        def __init__(self, config_path, log_dir, wandb_logger=None):
            super().__init__()
            self.config_path = config_path
            self.log_dir = log_dir
            self.wandb_logger = wandb_logger  # Store the logger

        def on_train_start(self, trainer, pl_module):
            # Only save on rank 0
            if trainer.global_rank == 0:
                config_dest = Path(self.log_dir) / "config_train.yaml"
                if self.config_path and Path(self.config_path).exists():
                    shutil.copy2(self.config_path, config_dest)
                    print(f"Saved config locally to: {config_dest}")

                    if self.wandb_logger:
                        try:
                            # self.wandb_logger.experiment is the wandb.Run object
                            # .save() uploads the file to the run's file directory
                            self.wandb_logger.experiment.save(self.config_path)
                            print(f"Saved {self.config_path} to wandb cloud.")
                        except Exception as e:
                            print(f"Error saving config to wandb: {e}")
                    else:
                        print("WandbLogger not provided, config not saved to cloud.")
                else:
                    print(f"Config file not found at {self.config_path}, cannot save.")

    config_callback = SaveConfigCallback(
        config_path="./config_train.yaml",  # Update with your config path
        log_dir=tensorboard_logger.log_dir,
        wandb_logger=wandb_logger,
    )
    # Train
    trainer = None
    if not args.skip_train:
        # Main training
        if not Path(
            TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            ).log_dir
        ).exists():
            trainer = Trainer(
                precision=args.precision,
                logger=logger_list,
                devices=args.gpus,
                accelerator="gpu",
                max_epochs=args.max_epochs,
                val_check_interval=0.5,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[
                    checkpoint_callback,
                    early_stop_callback,
                    lr_monitor_callback,
                    config_callback,
                    cm_callback,
                    expert_callback,
                    router_calibration_cb,
                ],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            if ckpt_path is not None:
                print(f"### Resume training from {ckpt_path}")

            trainer.fit(module, ckpt_path=ckpt_path)

            wandb.finish()

            try:
                os.chmod(tensorboard_logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

        if args.finetune:
            ckpt_path = sorted(  # load best model
                glob(f"{tensorboard_logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

            # Finetune trainer setting
            wandb_logger = WandbLogger(
                project="hier-egtr_family_classifier",
                name=f"{name}__finetune",
                group=name,
                save_dir="./logs",
                log_model=False,
            )
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = SGG(
                architecture=args.architecture,
                backbone_dirpath=args.backbone_dirpath,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone,
                lr_initialized=args.lr_initialized,
                weight_decay=args.weight_decay,
                pretrained=args.pretrained,
                main_trained=ckpt_path,
                from_scratch=args.from_scratch,
                id2label=id2label,
                rel_loss_coefficient=args.rel_loss_coefficient,
                smoothing=args.smoothing,
                rel_sample_negatives=args.rel_sample_negatives,
                rel_sample_nonmatching=args.rel_sample_nonmatching,
                rel_categories=rel_categories,
                multiple_sgg_evaluator=multiple_sgg_evaluator,
                multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
                single_sgg_evaluator=single_sgg_evaluator,
                single_sgg_evaluator_list=single_sgg_evaluator_list,
                coco_evaluator=coco_evaluator,
                oi_evaluator=oi_evaluator,
                feature_extractor=feature_extractor,
                num_queries=args.num_queries,
                ce_loss_coefficient=args.ce_loss_coefficient,
                rel_sample_negatives_largest=args.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
                use_freq_bias=args.use_freq_bias,
                fg_matrix=fg_matrix,
                use_log_softmax=args.use_log_softmax,
                freq_bias_eps=1e-12,
                connectivity_loss_coefficient=args.connectivity_loss_coefficient,
                logit_adjustment=args.logit_adjustment,
                logit_adj_tau=args.logit_adj_tau,
                hierarchical=args.hierarchical,
                num_geometric=args.num_geometric,
                num_possessive=args.num_possessive,
                num_semantic=args.num_semantic,
                num_negatives=args.num_negatives,
                super_weight=args.super_weight,
                train_relation_head=args.train_head,
                use_class_context=args.use_class_context,
            )

            # Finetune callback
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                filename="{epoch:02d}-{validation_loss:.2f}",
                save_last=True,
            )
            early_stop_callback = EarlyStopping(
                monitor="validation_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )
            lr_monitor_callback = LearningRateMonitor(logging_interval="step")

            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger_list,
                devices=args.gpus,
                accelerator="gpu",
                max_epochs=args.max_epochs_finetune,
                val_check_interval=0.5,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[
                    checkpoint_callback,
                    early_stop_callback,
                    lr_monitor_callback,
                    cm_callback,
                    expert_callback,
                    router_calibration_cb,
                ],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")

            trainer.fit(module, ckpt_path=None)

        if trainer is not None:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            try:
                os.chmod(tensorboard_logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

    # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        if args.skip_train and args.finetune:
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )

        # Load best model
        if args.load_model:
            # Get sorted list, select latest checkpoint file
            all_ckpts = glob(f"{args.load_model}/checkpoints/epoch=*.ckpt")
            assert all_ckpts, f"No checkpoints found in {args.load_model}/checkpoints"
            ckpt_path = sorted(
                all_ckpts, key=lambda x: int(x.split("epoch=")[1].split("-")[0])
            )[-1]
        else:
            ckpt_path = sorted(
                glob(f"{tensorboard_logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)  # load best model

        # Eval
        trainer = Trainer(
            precision=args.precision,
            logger=logger_list,
            devices=args.gpus,
            accelerator="gpu",
            max_epochs=-1,
        )
        if "visual_genome" in args.data_path:
            test_dataset = VGDataset(
                data_folder=args.data_path,
                feature_extractor=feature_extractor,
                split=args.split,
                num_object_queries=args.num_queries,
            )
        else:
            test_dataset = OIDataset(
                data_folder=args.data_path,
                split=args.split,
                num_object_queries=args.num_queries,
                feature_extractor=feature_extractor,
            )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=lambda x: collate_fn(x, feature_extractor),
            batch_size=args.eval_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=True,
        )
        if trainer.is_global_zero:
            print("### Evaluation")
        metric = trainer.test(module, dataloaders=test_dataloader)

        # Save eval metric
        metric = metric[0]
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"
        metric["eval_arg"] = args.__dict__
        with open(f"{filename}.json", "w") as f:
            json.dump(metric, f)
        print("metric is saved in", f"{filename}.json")
