# References:
# - https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
# - https://github.com/suprosanna/relationformer/blob/scene_graph/datasets/get_dataset_counts.py

import json
import os

import ipdb
import numpy as np
import torch
import torchvision
from tqdm import tqdm


class VGDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, data_folder, feature_extractor, split, debug=False, ann_file_name=None
    ):
        if ann_file_name is None:
            ann_file_name_to_load = f"{split}.json"
        else:
            ann_file_name_to_load = ann_file_name

        ann_file = os.path.join(data_folder, ann_file_name_to_load)
        img_folder = os.path.join(data_folder, "images")
        print(f"Loading COCO annotations from: {ann_file}")
        super(VGDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
        self.split = split
        self.debug = debug
        print(f"Loaded {len(self.ids)} image IDs from {ann_file_name_to_load}")

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(VGDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        target["class_labels"] -= 1  # remove 'no_relation' category
        return pixel_values, target

    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.ids)


class VGDataset(VGDetection):
    def __init__(
        self,
        data_folder,
        feature_extractor,
        split,
        num_object_queries=100,
        debug=False,
        relation_file_name="rel.json",
        ann_file_name=None
    ):
        super(VGDataset, self).__init__(data_folder, feature_extractor, split, debug, ann_file_name)
        relation_file_path = os.path.join(data_folder, relation_file_name)
        print(f"Loading relation annotations from: {relation_file_path}")
        try:
            with open(relation_file_path, "r") as f:
                rel = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Relation file not found at {relation_file_path}")

        if split not in rel:
            raise ValueError(
                f"Split '{split}' not found in relation file {relation_file_name}"
            )

        self.rel = rel[split]

        if "rel_categories" not in rel:
            raise ValueError(f"'rel_categories' key not found in {relation_file_name}")

        if rel["rel_categories"][0] == "__background__":
            self.rel_categories = rel["rel_categories"][1:]
            self.bg_class = True
        else:
            self.rel_categories = rel["rel_categories"]
            self.bg_class = False

        print(
            f"Loaded {len(self.rel_categories)} relation categories for this dataset."
        )

        self.num_object_queries = num_object_queries

        rel_image_ids = {int(k) for k in self.rel.keys()}
        coco_image_ids = set(self.ids)

        if coco_image_ids != rel_image_ids:
             print(f"Warning: Mismatch between image IDs ({len(coco_image_ids)}) and {relation_file_name} ({len(rel_image_ids)})!")
             print(f" -> IDs only in COCO file: {list(coco_image_ids - rel_image_ids)[:5]}...")
             print(f" -> IDs only in Relation file: {list(rel_image_ids - coco_image_ids)[:5]}...")
             assert 0

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(VGDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        rel_list = self.rel[str(image_id)]
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        rel = np.array(rel_list)

        target["rel"] = self._get_rel_tensor(rel)
        target["class_labels"] -= 1

        return pixel_values, target

    def _get_rel_tensor(self, rel_tensor):
        indices = rel_tensor.T
        num_family_rel = len(self.rel_categories)

        if self.bg_class:
            indices[-1, :] -= 1  # remove 'no_relation' category only if using rel.json

        rel = torch.zeros(
            [self.num_object_queries, self.num_object_queries, num_family_rel]
        )
        if torch.tensor(indices).numel() > 0:
            s_indices = torch.tensor(indices[0, :])
            o_indices = torch.tensor(indices[1, :])
            r_indices = torch.tensor(indices[2, :])

            valid_s = (s_indices >= 0) & (s_indices < self.num_object_queries)
            valid_o = (o_indices >= 0) & (o_indices < self.num_object_queries)
            valid_r = (r_indices >= 0) & (r_indices < num_family_rel)
            valid_mask = torch.logical_and(torch.logical_and(valid_s, valid_o), valid_r)

            if not torch.all(valid_mask):
                assert 0, "Out-of-bounds indices detected in _get_rel_tensor"

        # map each relation to original position
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0

        return rel


# https://github.com/suprosanna/relationformer/blob/75c24f61a81466df8f40c498e5f7aae3edd5ac6b/datasets/get_dataset_counts.py#L9
def vg_get_statistics(train_data, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data:
    :param must_overlap:
    :return:
    """
    num_classes = len(train_data.coco.cats)
    num_predicates = len(train_data.rel_categories)

    fg_matrix = np.zeros(
        (
            num_classes + 1,
            num_classes + 1,
            num_predicates,
        ),
        dtype=np.int64,
    )

    rel = train_data.rel
    for idx in tqdm(range(len(train_data))):
        image_id = train_data.ids[idx]

        if str(image_id) not in rel:
            assert 0,"image_id not in rel"# skip if this image has no relations for the specified family

        target = train_data.coco.loadAnns(train_data.coco.getAnnIds(image_id))
        gt_classes = np.array(list(map(lambda x: x["category_id"], target)))
        rel_list = rel[str(image_id)]

        if not rel_list:
            assert 0, "Should not contain empty relation list"

        gt_indices = np.array(torch.Tensor(rel_list).T, dtype="int64")
        gt_indices[-1, :] -= 1

        # foreground
        o1o2 = gt_classes[gt_indices[:2, :]].T
        for (o1, o2), gtr in zip(o1o2, gt_indices[2]):
            fg_matrix[o1 - 1, o2 - 1, gtr] += 1

    return fg_matrix
