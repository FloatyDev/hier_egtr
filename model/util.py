# Copied from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/detr/modeling_detr.py

from typing import List, Optional

import matplotlib as plt
import ipdb
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.box_ops import rescale_bboxes
import pytorch_lightning as pl
import os
import wandb
import torch
from torch import Tensor, nn


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# below: bounding box utilities taken from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.

    Returns:
        area (Tensor[N]): area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# below: taken from https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


def count_trainable(model, debugging=False):
    """
    Return total #trainable parameters.
    Used for debugging
    """
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(n_params for _, n_params in trainable)

    if debugging:
        print(f"trainable params: {total:,}  ({total/1e6:.2f} M)")

    return total


def get_super_rel_map():
    return [
        # 1-6: geometric
        0,  # 1 above -> geometric
        0,  # 2 accross -> geometric
        0,  # 3 against -> geometric
        0,  # 4 along -> geometric
        0,  # 5 and -> geometric
        0,  # 6 at -> geometric
        2,  # 7 attached to -> semantic
        0,  # 8 behind  -> geometric
        1,  # 9: belonging to -> possessive
        0,  # 10: between -> geometric
        2,  # 11 carrying -> semantic
        2,  # 12 covered in -> semantic
        2,  # 13 covering -> semantic
        2,  # 14 eating -> semantic
        2,  # 15 flying in -> semantic
        1,  # 16 for (misc) -> possessive (exclusion)
        1,  # 17 from (misc) -> possessive (exclusion)
        2,  # 18 growing on () -> semantic
        2,  # 19 hanging from -> semantic
        1,  # 20: has -> possessive
        2,  # 21: holding -> semantic (light_semantic_posession treated as semantic)
        0,  # 22 in -> geometric
        0,  # 23 in front of -> geometric
        2,  # 24 laying on -> semantic
        2,  # 25 looking at -> semantic
        2,  # 26 lying on -> semantic
        1,  # 27 made of -> possessive
        2,  # 28: mounted on -> semantic
        0,  # 29: near -> geometric
        1,  # 30: of -> possessive
        0,  # 31 on -> geometric
        0,  # 32 on back of -> geometric
        0,  # 33 over -> geometric
        # 34-35: semantic
        2,  # 34 painted on -> semantic
        2,  # 3 parked on -> semantic
        1,  # 36: part of -> possessive
        2,  # 37 playing -> semantic
        2,  # 38 riding -> semantic
        2,  # 39 says -> semantic
        2,  # 40 sitting on -> semantic
        2,  # 41 standing on -> semantic
        1,  # 42: to -> possessive
        0,  # 43: under -> geometric
        2,  # 44: using -> semantic (light_semantic_posession treated as semantic)
        2,  # 45 walking in -> semantic
        2,  # 46 walking on -> semantic
        2,  # 47 watching -> semantic
        1,  # 48 wearing -> possessive
        1,  # 49 wears -> possessive
        1,  # 50: with -> possessive
    ]


def get_orig2idx():
    fam_lists = {0: [], 1: [], 2: []}
    super_map = get_super_rel_map()
    for r, f in enumerate(super_map):
        fam_lists[f].append(r)

    orig2famidx = torch.full((len(super_map),), -1, dtype=torch.long)

    for f in (0, 1, 2):
        for j, orig_id in enumerate(fam_lists[f]):
            orig2famidx[orig_id] = j

    num_geometric = len(fam_lists[0])
    num_possessive = len(fam_lists[1])
    num_semantic = len(fam_lists[2])

    return orig2famidx, num_geometric, num_possessive, num_semantic


class GTTripletVis(pl.Callback):
    """
    Plot GT triplets straight from the batch that flows through training.

    Parameters
    ----------
    dataset          : VGDataset      (training split) – to resolve img paths
    id2label         : dict[int,str]  – object id ➜ name
    rel_categories   : list[str]      – predicates
    freq             : int            – visualise every `freq` steps
    max_triplets     : int | None     – truncate long lists for readability
    dpi              : int            – Matplotlib DPI of the figure
    """

    def __init__(
        self,
        dataset,
        id2label,
        rel_categories,
        max_triplets=None,
        freq: int = 500,
        dpi: int = 120,
    ):
        super().__init__()
        self.dataset = dataset
        self.id2label = id2label
        self.rel_categories = rel_categories
        self.freq = freq
        self.max_triplets = max_triplets
        self.dpi = dpi

    # --------------------------------------------------------------------- #
    #                   Lightning hook: called every batch                  #
    # --------------------------------------------------------------------- #
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        global_step = trainer.global_step
        if global_step % self.freq != 0:  # skip most batches
            return

        if isinstance(batch, dict):  # batch_size == 1 case
            targets = batch["labels"]  # List[dict] len==1
        else:  # normal tuple from collate
            _, targets = batch  # targets already a list

        tgt = targets[0]
        image_id = int(tgt["image_id"].item())

        # ------------------------------------------------------------------ #
        meta = self.dataset.coco.imgs[image_id]
        img_file = meta["file_name"]
        if not os.path.isabs(img_file):
            img_file = os.path.join(self.dataset.root, img_file)

        import PIL.Image as Image

        image = Image.open(img_file).convert("RGB")
        W, H = image.size

        # ------------------------------------------------------------------ #
        # 2. Convert GT tensors ➜ numpy
        boxes_xyxy = rescale_bboxes(tgt["boxes"].cpu(), (W, H)).numpy()
        obj_cls = tgt["class_labels"].cpu().numpy()  # (N,)
        rel = tgt["rel"].cpu()  # (N,N,R)
        trip_idx = torch.nonzero(rel, as_tuple=False).cpu().numpy()

        if self.max_triplets is not None:
            trip_idx = trip_idx[: self.max_triplets]

        # ------------------------------------------------------------------ #
        # 3. Draw
        fig, ax = plt.subplots(figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi)
        ax.imshow(image)
        cmap = plt.cm.get_cmap("hsv", len(trip_idx) + 1)

        for t, (sub_i, obj_i, rel_id) in enumerate(trip_idx, start=1):
            colour = cmap(t)

            # subject
            x1, y1, x2, y2 = boxes_xyxy[sub_i]
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=colour,
                    facecolor="none",
                )
            )
            ax.text(
                x1,
                y1 - 4,
                self.id2label[int(obj_cls[sub_i])],
                fontsize=8,
                color=colour,
                weight="bold",
            )

            # object
            xo1, yo1, xo2, yo2 = boxes_xyxy[obj_i]
            ax.add_patch(
                patches.Rectangle(
                    (xo1, yo1),
                    xo2 - xo1,
                    yo2 - yo1,
                    linewidth=2,
                    edgecolor=colour,
                    facecolor="none",
                )
            )
            ax.text(
                xo1,
                yo1 - 4,
                self.id2label[int(obj_cls[obj_i])],
                fontsize=8,
                color=colour,
                weight="bold",
            )

            # predicate label at mid-point
            xm, ym = (x1 + x2 + xo1 + xo2) / 4, (y1 + y2 + yo1 + yo2) / 4
            ax.text(
                xm,
                ym,
                self.rel_categories[int(rel_id)],
                fontsize=8,
                color=colour,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

        ax.axis("off")
        fig.tight_layout()
        plt.draw()

        # ------------------------------------------------------------------ #
        # 4. Log the figure
        def _log_figure(fig, tag: str, step: int, trainer):
            """
            Send `fig` to every logger registered in the current Trainer.
            Works with Lightning >=1.6 where `trainer.logger` may be
            a LoggerCollection.
            """
            loggers = (
                trainer.loggers
                if hasattr(trainer, "loggers")  # pl>=2.0
                else (
                    trainer.logger
                    if isinstance(trainer.logger, (list, tuple))
                    else [trainer.logger]
                )
            )

            for lg in loggers:
                # --- TensorBoard ---------------------------------------------------
                if hasattr(lg, "experiment") and hasattr(lg.experiment, "add_figure"):
                    lg.experiment.add_figure(tag, fig, global_step=step)

                # --- WandB ---------------------------------------------------------
                if isinstance(lg, pl.loggers.WandbLogger):
                    lg.experiment.log({tag: wandb.Image(fig)}, step=step)

        # ------------------------------------------------------------------ #
        # … inside on_train_batch_end after creating `fig`
        _log_figure(fig, tag="GT_triplets/train", step=global_step, trainer=trainer)
        plt.close(fig)
