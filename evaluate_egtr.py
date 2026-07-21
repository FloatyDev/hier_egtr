# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.open_image import OIDataset
from data.visual_genome import VGDataset
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OIEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_f_at_k,
    calculate_mR_from_evaluator_list,
)
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from train_egtr import collate_fn, evaluate_batch, evaluate_batch_all_modes


@torch.no_grad()
def calculate_fps(model, dataloader):
    model.eval()
    for batch in tqdm(dataloader):
        outputs = model(
            pixel_values=batch["pixel_values"].cuda(),
            pixel_mask=batch["pixel_mask"].cuda(),
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )


# Reference: https://github.com/facebookresearch/detr/blob/main/engine.py
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    num_labels,
    matcher,
    eval_modes,
    single_sgg_evaluator=None,
    oi_evaluator=None,
    coco_evaluator=None,
    feature_extractor=None,
):
    metric_dict = {}
    model.eval()

    rel_categories = dataloader.dataset.rel_categories
    single_sgg_evaluator_list = []

    # Initialize per-category evaluators
    if single_sgg_evaluator is not None:
        for index, name in enumerate(rel_categories):
            single_sgg_evaluator_list.append(
                (
                    index,
                    name,
                    BasicSceneGraphEvaluator.all_modes(multiple_preds=False),
                )
            )

    print("Starting evaluation...")
    for batch in tqdm(dataloader):
        # Move batch to device
        batch_pixel = batch["pixel_values"].to(model.device)
        batch_mask = batch["pixel_mask"].to(model.device)

        outputs = model(
            pixel_values=batch_pixel,
            pixel_mask=batch_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )

        targets = batch["labels"]
        evaluate_batch_all_modes(
            outputs=outputs,
            targets=targets,
            matcher=matcher,
            eval_modes=eval_modes,
            num_obj_labels=num_labels,
            single_sgg_evaluator=single_sgg_evaluator,
            single_sgg_evaluator_list=(
                single_sgg_evaluator_list
            ),
            max_topk=100,
        )
        #evaluate_batch(
        #    outputs,
        #    targets,
        #    single_sgg_evaluator,
        #    single_sgg_evaluator_list,
        #    num_labels,
        #)

        if coco_evaluator is not None:
            orig_target_sizes = torch.stack(
                [target["orig_size"] for target in targets], dim=0
            ).to(model.device)

            results = feature_extractor.post_process(outputs, orig_target_sizes)

            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, results)
            }
            coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        metric_dict.update({"AP50": coco_evaluator.coco_eval["bbox"].stats[1]})

    if single_sgg_evaluator is not None:
        for mode in eval_modes:
            recall = single_sgg_evaluator[
                mode
            ].print_stats()

            mean_recall = calculate_mR_from_evaluator_list(
                single_sgg_evaluator_list,
                mode,
                multiple_preds=False,
            )

            for key, value in recall.items():
                metric_dict[
                    f"{mode}/single/{key}"
                ] = value

            for key, value in mean_recall.items():
                metric_dict[
                    f"{mode}/single/{key}"
                ] = value
    #if single_sgg_evaluator is not None:
    #    recall = single_sgg_evaluator["sgdet"].print_stats()

    #    mean_recall = calculate_mR_from_evaluator_list(
    #        single_sgg_evaluator_list,
    #        "sgdet",
    #        multiple_preds=False,
    #    )

    #    f_at_k = calculate_f_at_k(
    #        recall=recall,
    #        mean_recall=mean_recall,
    #    )

    #    # Convert all values to native Python floats for safe JSON serialization.
    #    recall_metrics = {
    #        f"(single){key}": float(value) for key, value in recall.items()
    #    }
    #    mean_recall_metrics = {
    #        f"(single){key}": float(value) for key, value in mean_recall.items()
    #    }
    #    f_at_k_metrics = {
    #        f"(single){key}": float(value) for key, value in f_at_k.items()
    #    }

    #    metric_dict.update(recall_metrics)
    #    metric_dict.update(mean_recall_metrics)
    #    metric_dict.update(f_at_k_metrics)

    if oi_evaluator is not None:
        metrics = oi_evaluator.aggregate_metrics()
        metric_dict.update(metrics)

    return metric_dict


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument(
        "--artifact_path",
        type=str,
        required=True,
    )

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)

    # Evaluation
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)
    parser.add_argument("--ckpt", type=str, default="")

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # FPS
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--infer_only", type=str2bool, default=False)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    # Hierarchical
    parser.add_argument("--hier", type=bool, default=False)
    args, unknown = parser.parse_known_args()  # to ignore args when training

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    # Dataset
    if "visual_genome" in args.data_path:
        test_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        id2label = {
            k - 1: v["name"] for k, v in test_dataset.coco.cats.items()
        }  # 0 ~ 149
        coco_evaluator = CocoEvaluator(
            test_dataset.coco, ["bbox"]
        )  # initialize evaluator with ground truths
        oi_evaluator = None
    elif "open-image" in args.data_path:
        test_dataset = OIDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        id2label = test_dataset.classes_to_ind  # 0 ~ 600
        oi_evaluator = OIEvaluator(
            test_dataset.rel_categories, test_dataset.ind_to_classes
        )
        coco_evaluator = None

    # Dataloader
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    if args.eval_single_preds:  # Use this flag for graph constraint evaluation
        singe_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

    # Model
    config = DeformableDetrConfig.from_pretrained(args.artifact_path)
    config.logit_adjustment = args.logit_adjustment
    config.logit_adj_tau = args.logit_adj_tau
    config.hierarchical = args.hier

    matcher = DeformableDetrHungarianMatcher(
        class_cost=config.ce_loss_coefficient,
        bbox_cost=config.bbox_cost,
        giou_cost=config.giou_cost,
        smoothing=config.smoothing,
    )

    model = DetrForSceneGraphGeneration.from_pretrained(
        args.architecture, config=config, ignore_mismatched_sizes=True
    )

    if args.ckpt:
        ckpt_to_load = args.ckpt
    else:
        assert (
            args.artifact_path
        ), "--artifact_path is required when a ckpt_path is not given explicitely"
        ckpt_to_load = sorted(
            glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]

    state_dict = torch.load(ckpt_to_load, map_location="cpu")["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    model.cuda()

    # FPS
    if args.infer_only:
        calculate_fps(model, test_dataloader)
    # Eval
    else:
        metric = evaluate(
            model,
            test_dataloader,
            max(id2label.keys()) + 1,
            single_sgg_evaluator,
            oi_evaluator,
            coco_evaluator,
            feature_extractor,
        )
        #metric = evaluate(
        #    model,
        #    test_dataloader,
        #    max(id2label.keys()) + 1,
        #    singe_sgg_evaluator,
        #    oi_evaluator,
        #    coco_evaluator,
        #    feature_extractor,
        #)

        # Save eval metric
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_to_load.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"

        metric["eval_arg"] = args.__dict__
        metric_path = f"{filename}.json"

        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(
                metric,
                f,
                indent=4,
                allow_nan=False,
            )
        print("metric is saved in", metric_path)
        print("metric is saved in", f"{filename}.json")
