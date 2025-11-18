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
)
from model.egtr import DetrForSceneGraphGeneration
from util.box_ops import rescale_bboxes
from util.misc import use_deterministic_algorithms
from model.util import GTTripletVis, count_trainable, get_super_rel_map, get_orig2idx

seed_everything(42, workers=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")


def build_flat_pred_rel(geo, poss, sem, orig2fam, orig2famidx):
    N = geo.shape[0]
    num_rel_classes = len(orig2fam)
    pred_rel = torch.zeros((N, N, num_rel_classes), device=geo.device)

    orig2fam = torch.as_tensor(orig2fam)  # to do element-wise comparison

    fam_mask_geo = orig2fam == 0
    fam_mask_poss = orig2fam == 1
    fam_mask_sem = orig2fam == 2

    pred_rel[..., fam_mask_geo] = geo[..., orig2famidx[fam_mask_geo]]
    pred_rel[..., fam_mask_poss] = poss[..., orig2famidx[fam_mask_poss]]
    pred_rel[..., fam_mask_sem] = sem[..., orig2famidx[fam_mask_sem]]

    return pred_rel


# Reference: https://github.com/yuweihao/KERN/blob/master/models/eval_rels.py
def evaluate_batch(
    outputs,
    targets,
    multiple_sgg_evaluator,
    multiple_sgg_evaluator_list,
    single_sgg_evaluator,
    single_sgg_evaluator_list,
    oi_evaluator,
    num_labels,
    max_topk=100,
    hierarchical=True,
    partition_data=None,
):
    if hierarchical:
        if partition_data is not None:
            orig2fam = partition_data["super_rel_map"]
            orig2famidx = partition_data["orig2idx"]
        else:
            orig2fam = get_super_rel_map()
            orig2famidx = get_orig2idx()[0]

    for j, target in enumerate(targets):
        # Pred
        if hierarchical:
            geo, poss, sem, super, _ = outputs["pred_rel"]
            geo = geo[0].exp()
            poss = poss[0].exp()
            sem = sem[0].exp()
            super = super[0].exp()
            pred_rel = build_flat_pred_rel(geo, poss, sem, orig2fam, orig2famidx)
        else:
            pred_rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)

        pred_logits = outputs["logits"][j]
        obj_scores, pred_classes = torch.max(
            pred_logits.softmax(-1)[:, :num_labels], -1
        )
        sub_ob_scores = torch.outer(obj_scores, obj_scores)
        sub_ob_scores[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
        ] = 0.0  # prevent self-connection

        pred_boxes = outputs["pred_boxes"][j]
        if "pred_connectivity" in outputs:
            pred_connectivity = torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0)
            pred_rel = torch.mul(pred_rel, pred_connectivity)

        # GT
        orig_size = target["orig_size"]
        target_labels = target["class_labels"]  # [num_objs]
        target_boxes = target["boxes"]  # [num_objs, 4]

        target_rel = target["rel"].nonzero()  # [num_rels, 3(s, o, p)]

        gt_entry = {
            "gt_relations": target_rel.clone().numpy(),
            "gt_boxes": rescale_bboxes(target_boxes, torch.flip(orig_size, dims=[0]))
            .clone()
            .numpy(),
            "gt_classes": target_labels.clone().numpy(),
        }

        if multiple_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 3(s,o,p)]
            rel_scores = (
                pred_rel.cpu()
                .clone()
                .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
            )  # [pred_rels]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }
            multiple_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )

            for pred_id, _, evaluator_rel in multiple_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )

        if single_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 2(s,o)]
            rel_scores = (
                pred_rel.cpu().clone().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]]
            )  # [pred_rels, 50]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }
            single_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )
            for pred_id, _, evaluator_rel in single_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )

        if oi_evaluator is not None:  # OI evaluation, return all possible indicies
            sbj_obj_inds = torch.cartesian_prod(
                torch.arange(pred_logits.shape[0]), torch.arange(pred_logits.shape[0])
            )
            pred_scores = (
                pred_rel.cpu().clone().numpy().reshape(-1, pred_rel.size(-1))
            )  # (num_obj * num_obj, num_rel_classes)

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "sbj_obj_inds": sbj_obj_inds,  # for oi, (num_obj * num_obj, num_rel_classes)
                "pred_scores": pred_scores,  # for oi, (num_obj * num_obj, num_rel_classes)
            }
            oi_evaluator(gt_entry, pred_entry)


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
        use_class_context=False,
        partition_data=None,
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

        self.config = config
        self.partition_data = partition_data

        if config.from_scratch:
            assert backbone_dirpath
            self.model = DetrForSceneGraphGeneration(
                config=config, fg_matrix=fg_matrix, partition_data=self.partition_data
            )
            self.model.model.backbone.load_state_dict(
                torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
            )
            self.initialized_keys = []
        else:
            # Load trained object detector
            self.model, load_info = DetrForSceneGraphGeneration.from_pretrained(
                pretrained,
                config=config,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                fg_matrix=fg_matrix,
                partition_data=self.partition_data,
            )
            self.initialized_keys = load_info["missing_keys"] + [
                _key for _key, _, _ in load_info["mismatched_keys"]
            ]
        # only train relation_head
        if train_relation_head:
            if not main_trained:
                # load trained egtr weights for main training
                assert artifact_path, "have to give artifact_path"
                print(f"Loading checkpoint config from: {artifact_path}")

                ckpt_config = DeformableDetrConfig.from_pretrained(artifact_path)

                ckpt_is_hierarchical = getattr(ckpt_config, "hierarchical", False)
                print(
                    f"Checkpoint config loaded. Checkpoint is hierarchical: {ckpt_is_hierarchical}"
                )

                ckpt_path = sorted(
                    glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
                    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
                )[-1]
                state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

                for k in list(state_dict.keys()):
                    if (
                        k.startswith("model.rel_predictor.")
                        and not ckpt_is_hierarchical
                    ):
                        print(f"----deleting {k}")
                        del state_dict[k]
                    else:
                        state_dict[k[6:]] = state_dict.pop(k)  # "model."

                missing, unexpected = self.model.load_state_dict(
                    state_dict, strict=False
                )
                print("[sgg] missing keys:", missing)
                print("[sgg] unexpected keys:", unexpected)

            # disable all parameters and enable training only the relation head
            for p in self.model.parameters():
                p.requires_grad = False

            # enable the layers that must learn
            allow = (
                "rel_predictor.",  # hierarchical head
                # "proj_q",  # query projection
                # "proj_k",  # key projection
                # "final_sub_proj",  # keeps sub-object embeddings in sync
                # "final_obj_proj",  # keeps object embeddings in sync
                "rel_predictor_gate",  # tiny gate mlp, if you use it
            )

            for n, p in self.model.named_parameters():
                if n.startswith(allow):
                    p.requires_grad = True

            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            unexpected = [n for n in trainable if not n.startswith(allow)]

            count_trainable(model=self.model, debugging=True)

            assert not unexpected, (
                f"[sgg] unexpected trainable parameters:\n  {unexpected[:10]}… "
                f"(total {len(unexpected)})"
            )

            print(f"[sgg] trainable parameter count: {len(trainable)}")

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
        del outputs
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        # Log metrics directly with epoch aggregation
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"training_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []  # Initialize collection list

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        loss_dict["loss"] = loss

        self.validation_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        # aggregate metrics across batches
        for k in self.validation_step_outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in self.validation_step_outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)
        self.validation_step_outputs.clear()

    @rank_zero_only
    def on_train_start(self) -> None:
        if hasattr(self, "logger") and self.logger is not None:
            log_dir = self.logger.log_dir
        else:
            log_dir = os.getcwd()

        self.config.save_pretrained(log_dir)
        return super().on_train_start()

    def test_step(self, batch, batch_idx):
        # get the inputs
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
                partition_data=self.partition_data,
            )
            # eval OD
            if self.coco_evaluator is not None:
                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = self.feature_extractor.post_process(
                    outputs, orig_target_sizes.to(self.device)
                )  # convert outputs of model to COCO api
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                self.coco_evaluator.update(res)

    def on_test_epoch_end(self):
        log_dict = {}
        # log OD
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


def generate_partition_data(seed, num_total=50, num_geo=15, num_poss=11, num_sem=24):
    print(f"Generating random partition with seed {seed}")
    assert num_geo + num_poss + num_sem == num_total, "Partition sizes don't match"

    predicate_ids = list(range(num_total))
    rng = np.random.default_rng(seed)
    rng.shuffle(predicate_ids)

    geo_preds = sorted(predicate_ids[:num_geo])
    poss_preds = sorted(predicate_ids[num_geo : num_geo + num_poss])
    sem_preds = sorted(predicate_ids[num_geo + num_poss :])

    super_rel_map = [0] * num_total
    orig2idx = [0] * num_total

    for i, pred_id in enumerate(geo_preds):
        super_rel_map[pred_id] = 0
        orig2idx[pred_id] = i
    for i, pred_id in enumerate(poss_preds):
        super_rel_map[pred_id] = 1
        orig2idx[pred_id] = i
    for i, pred_id in enumerate(sem_preds):
        super_rel_map[pred_id] = 2
        orig2idx[pred_id] = i

    return {
        "super_rel_map": super_rel_map,
        "orig2idx": torch.tensor(orig2idx, dtype=torch.long),
        "num_geo": num_geo,
        "num_poss": num_poss,
        "num_sem": num_sem,
    }


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
    parser.add_argument(
        "--random_partition_seed",
        type=int,
        default=None,
        help="Seed for random partition generation. "
        "If None, uses the default manual partition.",
    )

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

    if args.random_partition_seed is not None:
        partition_data = generate_partition_data(
            seed=args.random_partition_seed,
            num_total=50,  # Assuming 50 total predicates for VG
            num_geo=args.num_geometric,
            num_poss=args.num_possessive,
            num_sem=args.num_semantic,
        )
    else:
        # Load the default manual partition
        print("Using default manual partition from model.util")
        orig2famidx, num_geo, num_poss, num_sem = get_orig2idx()
        partition_data = {
            "super_rel_map": get_super_rel_map(),
            "orig2idx": orig2famidx,
            "num_geo": num_geo,
            "num_poss": num_poss,
            "num_sem": num_sem,
        }
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
        project="hier-egtr", log_model=False, save_dir="./logs", name=name
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
        use_class_context=args.use_class_context,
        partition_data=partition_data,
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

    class SaveConfigCallback(Callback):
        def __init__(self, config_path, log_dir):
            self.config_path = config_path
            self.log_dir = log_dir

        def on_train_start(self, trainer, pl_module):
            # Only save on rank 0 to avoid race conditions in DDP
            if trainer.global_rank == 0:
                config_dest = Path(self.log_dir) / "config_train.yaml"
                shutil.copy2(self.config_path, config_dest)
                print(f"Saved config to: {config_dest}")

    config_callback = SaveConfigCallback(
        config_path="./config_train.yaml",  # Update with your config path
        log_dir=tensorboard_logger.log_dir,
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
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[
                    checkpoint_callback,
                    early_stop_callback,
                    lr_monitor_callback,
                    config_callback,
                ],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            if ckpt_path is not None:
                print(f"### Resume training from {ckpt_path}")
            trainer.fit(module, ckpt_path=ckpt_path)

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
                partition_data=partition_data,
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

        test_coco_evaluator=None
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
            test_coco_evaluator = CocoEvaluator(test_dataset.coco, ["bbox"])
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

        if test_coco_evaluator is not None:
            module.coco_evaluator = test_coco_evaluator

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
