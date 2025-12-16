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

from sklearn.metrics import confusion_matrix
import seaborn as sns


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


class SupConLossHierar(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLossHierar, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        mapping_list = get_super_rel_map()
        self.register_buffer("orig2fam", torch.tensor(mapping_list, dtype=torch.long))

    def get_parent_label(self, labels):
        """
        Maps fine-grained labels (0-49) to parent families (0-2)
        using the correct Visual Genome mapping.
        """
        valid_mask = (labels >= 0) & (labels < len(self.orig2fam))

        parent_labels = torch.zeros_like(labels)
        parent_labels[valid_mask] = self.orig2fam[labels[valid_mask]]

        return parent_labels

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [N_rels, Dim] or [N_rels, n_views, Dim]
            labels:   [N_rels]
        """
        device = features.device

        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")

            parent_labels = self.get_parent_label(labels.squeeze(1)).view(-1, 1)

            mask_same_parent = (
                torch.eq(parent_labels, parent_labels.T).float().to(device)
            )

            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        if labels is not None:
            mask_same_parent = mask_same_parent.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        if labels is not None:
            logits_mask = logits_mask * mask_same_parent

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2,
        alpha=None,
        reduction="mean",
        task_type="binary",
        num_classes=None,
    ):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if (
            task_type == "multi-class"
            and alpha is not None
            and isinstance(alpha, (list, torch.Tensor))
        ):
            assert (
                num_classes is not None
            ), "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
            )

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification."""
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = nn.functional.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


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
        # clip-text mapping
        # 2,
        # 2,
        # 2,
        # 2,
        # 2,
        # 2,
        # 0,
        # 0,
        # 2,
        # 2,
        # 0,
        # 0,
        # 0,
        # 0,
        # 0,
        # 2,
        # 2,
        # 0,
        # 0,
        # 2,
        # 0,
        # 2,
        # 0,
        # 1,
        # 0,
        # 1,
        # 0,
        # 0,
        # 2,
        # 2,
        # 2,
        # 0,
        # 2,
        # 0,
        # 0,
        # 2,
        # 0,
        # 1,
        # 2,
        # 1,
        # 0,
        # 2,
        # 2,
        # 0,
        # 0,
        # 0,
        # 0,
        # 0,
        # 0,
        # 2,
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


def get_hierarchical_counts(fg_matrix):
    """
    Aggregates fine-grained relation counts into Super-Relation families.

    Args:
        fg_matrix: Numpy array or Tensor of shape (Num_Obj, Num_Obj, 50).
                   Contains raw counts of relationships in the dataset.

    Returns:
        torch.Tensor: Shape (3,) containing [Count_Geometric, Count_Possessive, Count_Semantic]
    """

    fine_counts = torch.tensor(fg_matrix.sum(axis=(0, 1)), dtype=torch.float32)

    # Mapping shape: [50]
    mapping = torch.tensor(get_super_rel_map(), device=fine_counts.device)

    family_counts = torch.zeros(3, device=fine_counts.device)

    for i in range(3):
        mask = mapping == i
        # Sum the counts of all relations in this family
        family_counts[i] = fine_counts[mask].sum()

    return family_counts


def get_super_bce_bias(fg_matrix, eps=1e-8):
    """
    Computes log-probability bias for BCE.
    Unlike Softmax, this does not normalize across the 3 families.
    It normalizes by the total occurrences of the (Subj, Obj) pair.

    Returns:
        torch.Tensor: (Num_Classes, Num_Classes, 3)
    """
    mapping = get_super_rel_map()
    num_objs = fg_matrix.shape[0]

    super_counts = np.zeros((num_objs, num_objs, 3), dtype=np.float32)
    for r_idx, family_id in enumerate(mapping):
        super_counts[:, :, family_id] += fg_matrix[:, :, r_idx]

    # coompute the total count of the (S, O) pair appearing in the dataset
    total_pair_counts = fg_matrix.sum(axis=2, keepdims=True)  # [N, N, 1]

    probs = (super_counts + eps) / (total_pair_counts + eps)

    return torch.from_numpy(np.log(probs))


def get_super_frequency_bias(fg_matrix, eps=1e-12, use_log=True):

    map = get_super_rel_map()

    num_objs = fg_matrix.shape[0]
    super_matrix = np.zeros((num_objs, num_objs, 3), dtype=np.float32)

    for r_idx, family_id in enumerate(map):
        super_matrix[:, :, family_id] += fg_matrix[:, :, r_idx]

    denom = super_matrix.sum(axis=2, keepdims=True) + eps
    probs = (super_matrix + eps) / denom

    if use_log:
        probs = np.log(probs)

    return torch.from_numpy(probs)


def get_super_root_frequency_bias(fg_matrix, eps=1e-12):
    mapping = get_super_rel_map()
    num_objs = fg_matrix.shape[0]

    super_matrix = np.zeros((num_objs, num_objs, 3), dtype=np.float32)
    for r_idx, family_id in enumerate(mapping):
        super_matrix[:, :, family_id] += fg_matrix[:, :, r_idx]

    global_counts = super_matrix.sum(axis=(0, 1), keepdims=True)
    root_global_counts = np.sqrt(global_counts)

    balanced_scores = (super_matrix + eps) / (root_global_counts + eps)

    # create probability distribution
    denom = balanced_scores.sum(axis=2, keepdims=True) + eps
    probs = balanced_scores / denom

    return torch.from_numpy(np.log(probs))


def get_class_weights(fg_matrix):
    """
    Calculates class weights to counteract the Geometric bias.

    Args:
        fg_matrix: (Num_Obj, Num_Obj, 50) count matrix from dataset stats
        beta: (Optional) Smoothing factor for "Effective Number of Samples"
              closer to 1.0 = more aggressive re-balancing.

    Returns:
        torch.Tensor: Shape (3,) weights for CrossEntropyLoss
    """
    # Sum over subject/object dimensions -> [50]
    if hasattr(fg_matrix, "sum"):  # Handles both tensor and numpy
        fine_counts = torch.tensor(fg_matrix.sum(axis=(0, 1)), dtype=torch.float32)

    # Map to families [0, 1, 2]
    mapping = torch.tensor(get_super_rel_map(), device=fine_counts.device)
    family_counts = torch.zeros(3, device=fine_counts.device)

    for i in range(3):
        mask = mapping == i
        family_counts[i] = fine_counts[mask].sum()

    # Formula: n_samples / (n_classes * count_class)
    total_samples = family_counts.sum()
    n_classes = 3

    weights = total_samples / (n_classes * family_counts)

    # weights = torch.clamp(weights, max=10.0)

    return weights


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

class SuperRelationConfusionMatrix(pl.Callback):
    def __init__(self, id2label, device="cpu"):
        super().__init__()
        self.super_cats = ["Geometric", "Possessive", "Semantic"]
        # Ensure mapping is available
        self.orig2fam = torch.tensor(get_super_rel_map(), device=device)

        self.preds = []
        self.targets = []
        from model.deformable_detr import DeformableDetrHungarianMatcher

        self.matcher = DeformableDetrHungarianMatcher(
            class_cost=1.0, bbox_cost=5.0, giou_cost=2.0
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.preds = []
        self.targets = []
        self.orig2fam = self.orig2fam.to(pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        model_out = outputs["outputs"]
        targets = outputs["targets"]

        pred_rel = model_out["pred_rel"]

        if isinstance(pred_rel, dict):
            pred_logits = pred_rel["super"]
        else:
            pred_logits = pred_rel

        outputs_for_matcher = {
            "logits": model_out["logits"],
            "pred_boxes": model_out["pred_boxes"],
        }

        indices, _ = self.matcher(outputs_for_matcher, targets)

        for i, (src_idx, tgt_idx) in enumerate(indices):
            rel_logits_matched = pred_logits[i][src_idx][:, src_idx]
            tgt_rel_matrix = targets[i]["rel"][tgt_idx][:, tgt_idx]

            active_mask = tgt_rel_matrix.sum(dim=-1) > 0
            if not active_mask.any():
                continue

            active_logits = rel_logits_matched[active_mask]  # [K, num_classes]

            raw_preds = active_logits.argmax(dim=-1)  # [K]

            if active_logits.shape[-1] == 50:
                pred_fam = self.orig2fam[raw_preds]
            else:
                # If we have 3 classes (Raw Router), use directly
                pred_fam = raw_preds

            # Ground Truth Mapping
            active_tgts = tgt_rel_matrix[active_mask]
            gt_rel_idx = active_tgts.argmax(dim=-1)
            gt_fam = self.orig2fam[gt_rel_idx]

            self.preds.append(pred_fam.cpu())
            self.targets.append(gt_fam.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.preds:
            return
        import torch

        all_preds = torch.cat(self.preds).numpy()
        all_targets = torch.cat(self.targets).numpy()

        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=[0, 1, 2],
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.super_cats,
            yticklabels=self.super_cats,
            ax=ax,
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label (Fused System)")
        ax.set_title(f"Super-Relation Confusion Matrix (Epoch {trainer.current_epoch})")

        if trainer.logger:
            loggers = (
                trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
            )
            for logger in loggers:
                if isinstance(logger, pl.loggers.WandbLogger):
                    import wandb

                    logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        import torch

class ExpertDiagnosticsCallback(pl.Callback):
    """
    Diagnoses Expert health by decoupling them from the Router.
    Metrics:
    - Oracle Acc: Accuracy of the Expert on its own family (ignoring Router).
    - Logit Scale: Average magnitude of the Expert's raw logits (detects 'shouting').
    - Router Agmt: How often the Router agrees with the Ground Truth family.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.global2local_cpu, _, _, _ = get_orig2idx()
        self.super_map_cpu = get_super_rel_map()

        self.global2local = torch.tensor(self.global2local_cpu, device=device)
        self.super_map = torch.tensor(self.super_map_cpu, device=device)

        self.reset_stats()

    def reset_stats(self):
        # 0: Geo, 1: Poss, 2: Sem
        self.stats = {
            0: {"correct": 0, "total": 0, "logits": []},
            1: {"correct": 0, "total": 0, "logits": []},
            2: {"correct": 0, "total": 0, "logits": []},
        }
        self.router_correct = 0
        self.total_samples = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_stats()
        # Ensure tensors are on the correct device
        if self.global2local.device != pl_module.device:
            self.global2local = self.global2local.to(pl_module.device)
            self.super_map = self.super_map.to(pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        model_out = outputs["outputs"]
        targets = outputs["targets"]
        pred_rel = model_out["pred_rel"]

        if not isinstance(pred_rel, dict):
            return

        experts = {0: pred_rel["geo"], 1: pred_rel["poss"], 2: pred_rel["sem"]}
        router_logits = pred_rel["super"]  # (B, N, N, 3)

        for i, target in enumerate(targets):
            tgt_rel = target["rel"]  # (N, N, 50)
            active_idx = tgt_rel.nonzero(as_tuple=False)  # (K, 3) -> (sub, obj, cls)
            if active_idx.shape[0] == 0:
                continue

            sub, obj, gt_cls = active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]

            gt_fam = self.super_map[gt_cls]  # (K,) e.g., [0, 2, 2, 1...]
            gt_local = self.global2local[gt_cls]  # (K,) e.g., [3, 0, 5, 2...]

            r_logits = router_logits[i, sub, obj]  # (K, 3)
            r_preds = r_logits.argmax(dim=-1)
            self.router_correct += (r_preds == gt_fam).sum().item()
            self.total_samples += len(gt_fam)

            for fam_id in [0, 1, 2]:
                mask = gt_fam == fam_id
                if not mask.any():
                    continue

                curr_expert_logits = experts[fam_id][
                    i, sub[mask], obj[mask]
                ]  # (M, n_local)
                curr_gt_local = gt_local[mask]  # (M,)

                preds = curr_expert_logits.argmax(dim=-1)
                num_correct = (preds == curr_gt_local).sum().item()
                avg_logit = curr_expert_logits.max(dim=-1)[0].mean().item()

                self.stats[fam_id]["correct"] += num_correct
                self.stats[fam_id]["total"] += len(curr_gt_local)
                self.stats[fam_id]["logits"].append(avg_logit)

    def on_validation_epoch_end(self, trainer, pl_module):
        print("\n" + "=" * 40)
        print(f" EXPERT DIAGNOSTICS (Epoch {trainer.current_epoch})")
        print("=" * 40)

        # Router Stats
        r_acc = self.router_correct / max(self.total_samples, 1)
        print(f" [Router] Gate Accuracy: {r_acc:.2%}")

        families = ["Geometric", "Possessive", "Semantic"]
        for fam_id, name in enumerate(families):
            data = self.stats[fam_id]
            if data["total"] > 0:
                oracle_acc = data["correct"] / data["total"]
                logit_scale = np.mean(data["logits"])
                print(f" [{name} Expert]")
                print(f"   - Oracle Acc:   {oracle_acc:.2%} (Capability Ceiling)")
                print(f"   - Avg Logit:    {logit_scale:.2f}  (Loudness)")
            else:
                print(f" [{name} Expert] No samples.")
        print("=" * 40 + "\n")

        if trainer.logger and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log(
                {
                    f"oracle_acc/{families[i]}": self.stats[i]["correct"]
                    / max(1, self.stats[i]["total"])
                    for i in range(3)
                }
            )
