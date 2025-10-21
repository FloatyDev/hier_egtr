# This script is modified to evaluate a 4-model pipeline:
# 1. A hierarchical "super" model.
# 2. Three specialist "family" models (geo, poss, sem).
# It runs the super-model once, extracts shared features, and
# feeds them to the specialist heads for combined prediction.

import argparse
import json
from glob import glob

import ipdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.electra.modeling_electra import ElectraSelfAttention

from data.visual_genome import VGDataset
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration

# Import helper functions
from model.util import get_orig2idx, get_super_rel_map
from train_egtr import collate_fn, evaluate_batch


@torch.no_grad()
def load_specialist_predictor(path, use_class_context, device):
    """
    Loads a specialist (flat) model from a checkpoint and
    returns only its relation predictor head.
    """
    print(f"Loading specialist predictor from: {path}")

    # Load config, ensuring it's set to flat (non-hierarchical)
    config = DeformableDetrConfig.from_pretrained(path)
    config.hierarchical = False  # Specialists are flat models
    config.use_class_context = use_class_context

    # Load the full model structure
    model = DetrForSceneGraphGeneration(config=config)

    # Find the checkpoint file
    ckpt_to_load = sorted(
        glob(f"{path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]

    # Load state dict
    state_dict = torch.load(ckpt_to_load, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # "model."

    model.load_state_dict(state_dict)

    # Return only the predictor head, set to eval mode
    return model.rel_predictor.to(device).eval()


@torch.no_grad()
def evaluate_pipeline(
    model_super,
    predictor_geo,
    predictor_poss,
    predictor_sem,
    dataloader,
    num_labels,
    multiple_sgg_evaluator=None,
    single_sgg_evaluator=None,
    oi_evaluator=None,
    coco_evaluator=None,
    feature_extractor=None,
    orig2fam=None,
    orig2famidx=None,
    device=None,
):
    """
    Runs the full 4-model evaluation pipeline.
    """
    metric_dict = {}
    model_super.eval()

    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if multiple_sgg_evaluator is not None:
        for index, name in enumerate(dataloader.dataset.rel_categories):
            multiple_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
            )
    if single_sgg_evaluator is not None:
        for index, name in enumerate(dataloader.dataset.rel_categories):
            single_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=False))
            )

    # Get family indices for the final 50-relation tensor
    orig2fam_tensor = torch.as_tensor(orig2fam).to(device)
    geo_indices = (orig2fam_tensor == 0).nonzero().squeeze()
    poss_indices = (orig2fam_tensor == 1).nonzero().squeeze()
    sem_indices = (orig2fam_tensor == 2).nonzero().squeeze()

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        targets = [
            {k: v.cpu() for k, v in label.items()} for label in batch["labels"]
        ]  # Targets stay on CPU

        outputs = model_super(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )

        # Extract all common features
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        pred_connectivity = outputs.pred_connectivity

        # This is the critical feature tensor you exposed
        gated_features = outputs.gated_relation_source

        # outputs.pred_rel is (geo, poss, sem, super, hc)
        super_family_probs = outputs.pred_rel[3].softmax(-1)  # [B, N, N, 3]
        prob_geo = super_family_probs[..., 0:1]
        prob_poss = super_family_probs[..., 1:2]
        prob_sem = super_family_probs[..., 2:3]

        # trained with BCE, so we sigmoid the output
        geo_probs = predictor_geo(gated_features, logits).sigmoid()
        poss_probs = predictor_poss(gated_features, logits).sigmoid()
        sem_probs = predictor_sem(gated_features, logits).sigmoid()

        B, N, _, _ = geo_probs.shape
        num_total_rels = len(orig2fam)
        final_rel_probs = torch.zeros(B, N, N, num_total_rels, device=device)

        final_rel_probs[..., geo_indices] = geo_probs * prob_geo
        final_rel_probs[..., poss_indices] = poss_probs * prob_poss
        final_rel_probs[..., sem_indices] = sem_probs * prob_sem

        pipeline_outputs = {
            "logits": logits.cpu(),
            "pred_boxes": pred_boxes.cpu(),
            "pred_connectivity": pred_connectivity.cpu(),
            "pred_rel": final_rel_probs.cpu(),  # Use your combined probabilities
        }

        evaluate_batch(
            pipeline_outputs,
            targets,
            multiple_sgg_evaluator,
            multiple_sgg_evaluator_list,
            single_sgg_evaluator,
            single_sgg_evaluator_list,
            oi_evaluator,
            num_labels,
            hierarchical=False,  # IMPORTANT!
        )

        if coco_evaluator is not None:
            orig_target_sizes = torch.stack(
                [target["orig_size"] for target in targets], dim=0
            )
            # We must use the *original* 'outputs' object from model_super
            # as post_process expects the raw model output, not our dict
            results = feature_extractor.post_process(
                outputs, orig_target_sizes.to(model_super.device)
            )
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

    if multiple_sgg_evaluator is not None:
        recall = multiple_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
        )
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

    if single_sgg_evaluator is not None:
        recall = single_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            single_sgg_evaluator_list, "sgdet", multiple_preds=False
        )
        recall = {f"(single){key}": value for key, value in recall.items()}
        mean_recall = {f"(single){key}": value for key, value in mean_recall.items()}
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

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
        "--path_super",
        type=str,
        required=True,
        help="Path to the main hierarchical model artifact",
    )
    parser.add_argument(
        "--path_geo",
        type=str,
        required=True,
        help="Path to the specialist geometric model artifact",
    )
    parser.add_argument(
        "--path_poss",
        type=str,
        required=True,
        help="Path to the specialist possessive model artifact",
    )
    parser.add_argument(
        "--path_sem",
        type=str,
        required=True,
        help="Path to the specialist semantic model artifact",
    )

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)

    # Evaluation
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # FPS (keeping for compatibility, but pipeline won't be measured)
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--infer_only", type=str2bool, default=False)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--use_class_context",
        type=str2bool,
        default=False,
        help="Whether the models were trained with class context",
    )

    args, unknown = parser.parse_known_args()  # to ignore args when training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        coco_evaluator = CocoEvaluator(test_dataset.coco, ["bbox"])
        oi_evaluator = None
    else:
        raise ValueError()

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
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    if args.eval_multiple_preds:
        multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    if args.eval_single_preds:
        single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

    print(f"Loading super model from: {args.path_super}")
    config_super = DeformableDetrConfig.from_pretrained(args.path_super)
    config_super.hierarchical = True  # Super model IS hierarchical
    config_super.use_class_context = args.use_class_context
    config_super.logit_adjustment = args.logit_adjustment
    config_super.logit_adj_tau = args.logit_adj_tau

    model_super = DetrForSceneGraphGeneration(config=config_super)

    ckpt_to_load = sorted(
        glob(f"{args.path_super}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]

    state_dict = torch.load(ckpt_to_load, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # "model."

    missing, unexpected = model_super.load_state_dict(state_dict)
    print(
        f"✓ loaded super model {ckpt_to_load} "
        f"({len(unexpected)} unexpected • {len(missing)} missing)"
    )
    model_super.to(device)

    # 2. Load Specialist Predictor Heads
    predictor_geo = load_specialist_predictor(
        args.path_geo, args.use_class_context, device
    )
    predictor_poss = load_specialist_predictor(
        args.path_poss, args.use_class_context, device
    )
    predictor_sem = load_specialist_predictor(
        args.path_sem, args.use_class_context, device
    )

    orig2fam = get_super_rel_map()
    orig2famidx, _, _, _ = get_orig2idx()

    metric = evaluate_pipeline(
        model_super,
        predictor_geo,
        predictor_poss,
        predictor_sem,
        test_dataloader,
        max(id2label.keys()) + 1,
        multiple_sgg_evaluator,
        single_sgg_evaluator,
        oi_evaluator,
        coco_evaluator,
        feature_extractor,
        orig2fam,
        orig2famidx,
        device,
    )

    device_name = "".join(torch.cuda.get_device_name(0).split()[1:2])
    filename = f'{ckpt_to_load.replace(".ckpt", "")}__PIPELINE__{args.split}__{len(test_dataloader)}__{device_name}'
    if args.logit_adjustment:
        filename += f"__la_{args.logit_adj_tau}"
    metric["eval_arg"] = args.__dict__
    with open(f"{filename}.json", "w") as f:
        json.dump(metric, f)
    print("metric is saved in", f"{filename}.json")
