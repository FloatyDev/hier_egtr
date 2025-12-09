# coding=utf-8
# Original sources:
#  - https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/detr/modeling_detr.py
#  - https://github.com/huggingface/transformers/blob/01eb34ab45a8895fbd9e335568290e5d0f5f4491/src/transformers/models/deformable_detr/modeling_deformable_detr.py

# Original code copyright
# Copyright 2021 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright
# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import ipdb
import torch
import torch.nn.functional as F
from torch import nn
from transformers.image_transforms import center_to_corners_format
from transformers.utils import ModelOutput
from model.util import (
    get_super_rel_map,
    get_orig2idx,
    get_super_frequency_bias,
    get_super_root_frequency_bias,
)

from .deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrPreTrainedModel,
    inverse_sigmoid,
)
from .util import (
    dice_loss,
    generalized_box_iou,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)


@dataclass
class DetrSceneGraphGenerationOutput(ModelOutput):
    """
    Output type of [`DetrForSceneGraphGeneration`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrFeatureExtractor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    pred_rel: Optional[torch.FloatTensor] = None
    pred_connectivity: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DualHeadRelationClassifier(nn.Module):
    """
    Teacher-Student Classifier optimized for Distillation.
    Replicates the DeformableDetrMLPPredictionHead structure to allow
    weight loading from a Flat-50 checkpoint.
    """

    def __init__(
        self,
        input_dim=512,  # 2 * d_model
        hidden_dim=256,  # d_model
        num_fine_classes=50,  # Teacher output (Matches Flat-50 output_dim)
        num_super_classes=3,  # Student output (Geometric, Possessive, Semantic)
        num_layers=3,  # Depth of the original Flat-50 MLP
    ):
        super(DualHeadRelationClassifier, self).__init__()

        # --- Shared Layers (The Frozen Feature Extractor) ---
        self.shared_layers = nn.ModuleList()

        self.shared_layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fine_head = nn.Linear(hidden_dim, num_fine_classes)

        # This is the only layer you will train.
        self.super_head = nn.Linear(hidden_dim, num_super_classes)

    def forward(self, features):
        # features: [Batch, Num_Queries, Num_Queries, Input_Dim]
        # Ignore det_logits, etc. as Flat-50 didn't use them.

        x = features

        # Pass through Shared Layers (with ReLU)
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        # Teacher Output (Frozen Features)
        logits_fine = self.fine_head(x)

        # Student Output (New Task)
        logits_super = self.super_head(x)

        return logits_fine, logits_super


class BayesianRelationClassifier(nn.Module):
    """
    Modified hierarchical classifier for 4D gated_relation_source features
    Maintains (bsz, num_queries, num_queries) structure
    """

    def __init__(
        self,
        input_dim=512,  # Should match 2 * d_model
        num_classes=150,
        num_geometric=15,
        num_possessive=11,
        num_semantic=24,
        class_embed_dim=64,
        T1=1,
        T2=1,
        T3=1,
        use_class_context=False,
    ):
        super(BayesianRelationClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_class_context = use_class_context
        if use_class_context:
            self.class_embed = nn.Embedding(num_classes, class_embed_dim)
            fused_dim = input_dim + 2 * class_embed_dim
        else:
            fused_dim = input_dim

        # Prediction heads
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 512),  # 512 + class emb
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Relationship heads
        self.fc3_1 = nn.Linear(512, num_geometric)
        self.fc3_2 = nn.Linear(512, num_possessive)
        self.fc3_3 = nn.Linear(512, num_semantic)
        self.fc5 = nn.Linear(512, 3)  # Super relation

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(
        self,
        features,  # gated_relation_source: (bsz, N, N, feat_dim)
        det_logits,  # class logits: (bsz, N, num_classes)
    ):
        B, N, _, D = features.shape

        assert features.dim() == 4, f"Expected 4D tensor, got {features.dim()}D"
        assert features.shape[1] == features.shape[2], "Expected square relation matrix"

        if self.use_class_context:
            with torch.no_grad():
                class_prob = det_logits.softmax(-1)
            emb = self.class_embed.weight

            soft_emb = torch.matmul(class_prob, emb)  # (bsz,N,64)
            ## Embed class information
            c1_emb = soft_emb.unsqueeze(2).expand(B, N, N, -1)  # subject
            c2_emb = soft_emb.unsqueeze(1).expand(B, N, N, -1)  # object
            ## Combine features and embeddings
            combined = torch.cat(
                [features, c1_emb, c2_emb], dim=-1
            )  # (bsz, N, N, 512+2*class_embed_dim)
        else:
            combined = features

        if (
            freq_bias is not None
            and subj_classes is not None
            and obj_classes is not None
        ):
            s_idx = subj_classes.unsqueeze(2).expand(B, N, N)
            o_idx = obj_classes.unsqueeze(1).expand(B, N, N)
            batch_bias = freq_bias[s_idx, o_idx]
            super_relation = super_relation + batch_bias

        # Compute outputs
        super_relation = F.log_softmax(self.fc5(hc), dim=-1)  # (bsz, N, N, 3)

        # Compute hierarchical relationships
        relation_1 = F.log_softmax(self.fc3_1(hc) / self.T1, dim=-1) + super_relation[
            ..., 0
        ].unsqueeze(
            -1
        )  # geo

        relation_2 = F.log_softmax(self.fc3_2(hc) / self.T2, dim=-1) + super_relation[
            ..., 1
        ].unsqueeze(
            -1
        )  # poss

        relation_3 = F.log_softmax(self.fc3_3(hc) / self.T3, dim=-1) + super_relation[
            ..., 2
        ].unsqueeze(
            -1
        )  # sem

        return (
            relation_1,  # (bsz, N, N, num_geometric)
            relation_2,  # (bsz, N, N, num_possessive)
            relation_3,  # (bsz, N, N, num_semantic)
            super_relation,  # (bsz, N, N, 3)
            hc,  # (bsz, N, N, 512) - intermediate features
        )


class DetrForSceneGraphGeneration(DeformableDetrPreTrainedModel):
    def __init__(self, config, partition_data=None, **kwargs):
        super(DetrForSceneGraphGeneration, self).__init__(config)
        self.model = DeformableDetrModel(config)

        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )
        # Initialization to stabilize training
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        )
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # hack implementation for two-stage
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.num_queries = self.config.num_queries
        self.head_dim = config.d_model // config.num_attention_heads
        self.layer_head = self.config.decoder_layers * config.num_attention_heads

        # Frequency bias initialization
        if kwargs.get("fg_matrix", None) is not None:  # when training
            eps = config.freq_bias_eps
            fg_matrix = kwargs.get("fg_matrix", None)
            self.fg_matrix = fg_matrix
            rel_dist = torch.FloatTensor(
                (fg_matrix.sum(axis=(0, 1))) / (fg_matrix.sum() + eps)
            )
            triplet_dist = torch.FloatTensor(
                fg_matrix + eps / (fg_matrix.sum(2, keepdims=True) + eps)
            )
            if config.use_log_softmax:
                triplet_dist = F.log_softmax(triplet_dist, dim=-1)
            else:
                triplet_dist = triplet_dist.log()
            self.rel_dist = nn.Parameter(rel_dist, requires_grad=False)
            self.triplet_dist = nn.Parameter(triplet_dist, requires_grad=False)
            del rel_dist, triplet_dist
        else:  # when infer
            self.triplet_dist = nn.Parameter(
                torch.Tensor(
                    config.num_labels + 1, config.num_labels + 1, config.num_rel_labels
                ),
                requires_grad=False,
            )
            self.rel_dist = nn.Parameter(
                torch.Tensor(config.num_rel_labels), requires_grad=False
            )

        self.proj_q = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for i in range(self.config.decoder_layers)
            ]
        )
        self.proj_k = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for i in range(self.config.decoder_layers)
            ]
        )
        self.final_sub_proj = nn.Linear(config.d_model, config.d_model)
        self.final_obj_proj = nn.Linear(config.d_model, config.d_model)

        self.rel_predictor_gate = nn.Linear(2 * config.d_model, 1)

        if self.config.hierarchical:
            self.rel_predictor = DualHeadRelationClassifier(
                input_dim=2 * config.d_model,
                hidden_dim=config.d_model,
                num_fine_classes=50,
                num_super_classes=3,
                num_layers=3
            )
        else:
            self.rel_predictor = DeformableDetrMLPPredictionHead(
                input_dim=2 * config.d_model,
                hidden_dim=config.d_model,
                output_dim=config.num_rel_labels,
                num_layers=3,
            )

        self.connectivity_layer = DeformableDetrMLPPredictionHead(
            input_dim=2 * config.d_model,
            hidden_dim=config.d_model,
            output_dim=1,
            num_layers=3,
        )
        self.partition_data = partition_data

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # outputs_class = outputs_class.transpose(1, 0)
        # outputs_coord = outputs_coord.transpose(1, 0)
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        output_attention_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_attention_states=output_attention_states,
            return_dict=return_dict,
        )

        sequence_output = outputs["last_hidden_state"]
        bsz = sequence_output.size(0)
        hidden_states = (
            outputs.intermediate_hidden_states if return_dict else outputs[2]
        )
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = (
            outputs.intermediate_reference_points if return_dict else outputs[3]
        )

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[level](hidden_states[:, level])
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(
                    f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}"
                )
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        del hidden_states, init_reference, inter_references
        # Keep batch_size as first dimension
        outputs_class = torch.stack(outputs_classes, dim=1)
        outputs_coord = torch.stack(outputs_coords, dim=1)
        del outputs_classes, outputs_coords

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        if self.config.auxiliary_loss:
            outputs_class = outputs_class[:, : self.config.decoder_layers, ...]
            outputs_coord = outputs_coord[:, : self.config.decoder_layers, ...]
            outputs_class = outputs_class.permute(1, 0, 2, 3)
            outputs_coord = outputs_coord.permute(1, 0, 2, 3)

        _, num_object_queries, _ = logits.shape
        unscaling = self.head_dim**0.5

        # Get self-attention byproducts from deformable detr
        decoder_attention_queries = outputs[
            "decoder_attention_queries"
        ]  # tuple of [bsz, num_heads, num_object_queries, d_head]
        outputs["decoder_attention_queries"] = None
        decoder_attention_keys = outputs[
            "decoder_attention_keys"
        ]  # tuple of [bsz, num_heads, num_object_queries, d_head]
        outputs["decoder_attention_keys"] = None

        # Unscaling & stacking attention queries
        projected_q = []
        for q, proj_q in zip(decoder_attention_queries, self.proj_q):
            projected_q.append(
                proj_q(
                    q.transpose(1, 2).reshape(
                        [bsz, num_object_queries, self.config.d_model]
                    )
                    * unscaling
                )
            )
        decoder_attention_queries = torch.stack(
            projected_q, -2
        )  # [bsz, num_object_queries, num_layers, d_model]
        del projected_q

        # Stacking attention keys
        projected_k = []
        for k, proj_k in zip(decoder_attention_keys, self.proj_k):
            projected_k.append(
                proj_k(
                    k.transpose(1, 2).reshape(
                        [bsz, num_object_queries, self.config.d_model]
                    )
                )
            )
        decoder_attention_keys = torch.stack(
            projected_k, -2
        )  # [bsz, num_object_queries, num_layers, d_model]
        del projected_k

        # Pairwise concatenation
        decoder_attention_queries = decoder_attention_queries.unsqueeze(2).repeat(
            1, 1, num_object_queries, 1, 1
        )
        decoder_attention_keys = decoder_attention_keys.unsqueeze(1).repeat(
            1, num_object_queries, 1, 1, 1
        )
        relation_source = torch.cat(
            [decoder_attention_queries, decoder_attention_keys], dim=-1
        )  # [bsz, num_object_queries, num_object_queries, num_layers, 2*d_model]
        del decoder_attention_queries, decoder_attention_keys

        # Use final hidden representations
        subject_output = (
            self.final_sub_proj(sequence_output)
            .unsqueeze(2)
            .repeat(1, 1, num_object_queries, 1)
        )
        object_output = (
            self.final_obj_proj(sequence_output)
            .unsqueeze(1)
            .repeat(1, num_object_queries, 1, 1)
        )
        del sequence_output
        relation_source = torch.cat(
            [
                relation_source,
                torch.cat([subject_output, object_output], dim=-1).unsqueeze(-2),
            ],
            dim=-2,
        )
        del subject_output, object_output

        # Gated sum
        rel_gate = torch.sigmoid(self.rel_predictor_gate(relation_source))
        gated_relation_source = torch.mul(rel_gate, relation_source).sum(dim=-2)

        if self.config.hierarchical:
            # Hierarchical prediction uses (log softmax)
            predicted_node = torch.argmax(logits, dim=-1)
            if self.config.use_freq_bias:
                pred_rel = self.rel_predictor(
                    gated_relation_source,
                    logits,
                    subj_classes=predicted_node,
                    obj_classes=predicted_node,
                    freq_bias=self.super_freq_bias,
                )
            else:
                pred_rel = self.rel_predictor(
                    gated_relation_source
                )

        else:
            # Original flat prediction
            pred_rel = self.rel_predictor(gated_relation_source)
        # Connectivity
        pred_connectivity = self.connectivity_layer(gated_relation_source)

        # from <Neural Motifs>
        if self.config.use_freq_bias and not self.config.hierarchical:
            predicted_node = torch.argmax(logits, dim=-1)
            pred_rel += torch.stack(
                [
                    self.triplet_dist[predicted_node[i]][:, predicted_node[i]]
                    for i in range(len(predicted_node))
                ],
                dim=0,
            )

        del gated_relation_source
        del relation_source

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DeformableDetrHungarianMatcher(
                class_cost=self.config.ce_loss_coefficient,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
                smoothing=self.config.smoothing,
            )  # the same as loss coefficients
            # Second: create the criterion
            losses = ["labels", "boxes", "relations", "cardinality", "uncertainty"]
            criterion = SceneGraphGenerationLoss(
                matcher=matcher,
                num_object_queries=num_object_queries,
                num_classes=self.config.num_labels,
                num_rel_labels=self.config.num_rel_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
                smoothing=self.config.smoothing,
                rel_sample_negatives=self.config.rel_sample_negatives,
                rel_sample_nonmatching=self.config.rel_sample_nonmatching,
                model_training=self.training,
                focal_alpha=self.config.focal_alpha,
                rel_sample_negatives_largest=self.config.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=self.config.rel_sample_nonmatching_largest,
                hierarchical=self.config.hierarchical,
                super_weight=self.config.super_weight,
                fg_matrix=self.fg_matrix,
                partition_data=self.partition_data,
            )

            criterion.to(self.device)

            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss["pred_rel"] = pred_rel
            outputs_loss["pred_connectivity"] = pred_connectivity

            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            if self.config.two_stage:
                enc_outputs_coord = outputs.enc_outputs_coord_logits.sigmoid()
                outputs_loss["enc_outputs"] = {
                    "logits": outputs.enc_outputs_class,
                    "pred_boxes": enc_outputs_coord,
                }

            loss_dict, hier_rel_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_ce": self.config.ce_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
            }
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict["loss_rel"] = self.config.rel_loss_coefficient
            weight_dict["loss_connectivity"] = self.config.connectivity_loss_coefficient
            aux_weight_dict = {}
            if self.config.auxiliary_loss:
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update(
                        {f"{k}_{i}": v for k, v in weight_dict.items()}
                    )

            two_stage_weight_dict = {}
            if self.config.two_stage:
                two_stage_weight_dict = {f"{k}_enc": v for k, v in weight_dict.items()}
            weight_dict.update(aux_weight_dict)
            weight_dict.update(two_stage_weight_dict)

            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            # rel_gate: [bsz, num_objects, num_objects, layer, 1]
            rel_gate = rel_gate.reshape(
                bsz * num_object_queries * num_object_queries, -1
            ).mean(0)
            log_layers = list(
                range(self.config.decoder_layers + 1)
            )  # include final layers

            for i, v in zip(log_layers, rel_gate):
                loss_dict[f"rel_gate_{i}"] = v
            if hier_rel_dict:
                loss_dict.update(hier_rel_dict)
        # from <structured sparse rcnn>, post-hoc logit adjustment.
        # reference: https://github.com/google-research/google-research/blob/master/logit_adjustment/main.py#L136-L140
        if self.config.logit_adjustment and not self.hierarchical:
            pred_rel = pred_rel - self.config.logit_adj_tau * self.rel_dist.log().to(
                pred_rel.device
            )

        pred_connectivity = pred_connectivity.sigmoid()
        if not self.config.hierarchical:
            pred_rel = pred_rel.sigmoid()

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrSceneGraphGenerationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_rel=pred_rel,
            pred_connectivity=pred_connectivity,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


def inspect_bias_statistics(freq_bias_tensor):
    """
    Analyzes the pre-computed frequency bias tensor.
    Args:
        freq_bias_tensor: torch.Tensor of shape (Num_Classes, Num_Classes, 3)
    """
    print("\n--- Bias Statistics Inspection ---")

    # Check for -inf (which happens if log(0) occurred)
    if torch.isinf(freq_bias_tensor).any():
        print(
            "WARNING: Bias contains -inf values! These will kill gradients immediately."
        )
        # Fix: Replace -inf with a large negative number that is safe for sigmoid
        # -10.0 results in sigmoid(-10) ~= 4.5e-5, which is small but allows gradient flow.
        min_safe_val = -10.0
        print(f"Suggestion: Clamp bottom range to {min_safe_val}")

    max_val = freq_bias_tensor.max().item()
    min_val = freq_bias_tensor.min().item()
    mean_val = freq_bias_tensor.mean().item()

    print(f"Max Log-Prob: {max_val:.4f}")
    print(f"Min Log-Prob: {min_val:.4f}")
    print(f"Mean Log-Prob: {mean_val:.4f}")

    # Check "Active Zone" Percentage (values between -4 and +4)
    in_range = (
        (freq_bias_tensor > -4.0) & (freq_bias_tensor < 4.0)
    ).float().mean() * 100
    print(
        f"Percentage of bias values in 'Gradient Safe Zone' (-4 to 4): {in_range:.2f}%"
    )
    print("----------------------------------\n")


def analyze_bias_distribution(bias_tensor, title="Bias Distribution"):
    """
    Plots a histogram of the bias values to check for saturation.
    """
    # Flatten tensor to 1D array
    values = bias_tensor.cpu().numpy().flatten()

    # Statistics
    min_v, max_v = values.min(), values.max()
    mean_v, std_v = values.mean(), values.std()

    # Percentage in Safe Zone (-4 to +4)
    safe_zone = ((values > -4) & (values < 4)).sum() / len(values) * 100

    print(f"--- {title} ---")
    print(f"Range: [{min_v:.2f}, {max_v:.2f}]")
    print(f"Mean:  {mean_v:.2f} +/- {std_v:.2f}")
    print(f"Gradient Safe Zone (-4 to +4): {safe_zone:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=100, color="blue", alpha=0.7)

    # Draw "Safe Zone" lines
    plt.axvline(x=-4, color="r", linestyle="--", label="Gradient Decay Start")
    plt.axvline(x=-6, color="k", linestyle="--", label="Gradient Death")

    plt.title(f"{title}\nSafe Zone: {safe_zone:.1f}%")
    plt.xlabel("Logit Bias Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save or Show
    plt.savefig("bias_distribution_check.png")
    print("Plot saved to bias_distribution_check.png")
    plt.close()


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class SceneGraphGenerationLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        num_object_queries,
        num_classes,
        num_rel_labels,
        eos_coef,
        losses,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        model_training,
        focal_alpha,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
        # Add hierarchical parameters
        fg_matrix,
        hierarchical=False,
        super_weight=1.0,  # weight of super relation at general rel loss sum
        partition_data=None,
    ):
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-6config.num_rel_labels269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_object_queries = num_object_queries
        self.num_classes = num_classes
        self.num_rel_labels = num_rel_labels
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.rel_sample_negatives = rel_sample_negatives
        self.rel_sample_nonmatching = rel_sample_nonmatching
        self.model_training = model_training
        self.focal_alpha = focal_alpha
        self.rel_sample_negatives_largest = rel_sample_negatives_largest
        self.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest
        self.nonmatching_cost = (
            -torch.log(torch.tensor(1e-8)) * matcher.class_cost
            + 4 * matcher.bbox_cost
            + 2 * matcher.giou_cost
            - torch.log(torch.tensor((1.0 / smoothing) - 1.0))
        )  # set minimum bipartite matching costs for nonmatched object queries

        self.hierarchical = hierarchical

        if self.hierarchical:
            if partition_data is None:
                print(
                    "WARNING: Hierarchical loss using default partition from model.util"
                )
                from model.util import get_super_rel_map, get_orig2idx

                self.super_relation_map = get_super_rel_map()
                orig2famidx, num_geo, num_poss, num_sem = get_orig2idx()
            else:
                self.super_relation_map = partition_data["super_rel_map"]
                orig2famidx = partition_data["orig2idx"]
                num_geo = partition_data["num_geo"]
                num_poss = partition_data["num_poss"]
                num_sem = partition_data["num_sem"]
        else:
            # Still need to define this for the non-hierarchical case if it's used
            from model.util import get_super_rel_map

            self.super_relation_map = get_super_rel_map()

        if hierarchical:

            self.register_buffer(
                "orig2fam",
                torch.tensor(self.super_relation_map, dtype=torch.long),
                persistent=True,
            )
            super_counts = get_hierarchical_counts(fg_matrix=fg_matrix)
            total = super_counts.sum()
            eps = 1e-12

            weights = total / (super_counts + eps)

            self.super_loss = nn.CrossEntropyLoss(reduction="none")
        else:
            # Original BCEWithLogitsLoss for flat mode
            self.rel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.connectivity_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss_labels(self, outputs, targets, indices, matching_costs, num_boxes):
        return self._loss_labels_focal(
            outputs, targets, indices, matching_costs, num_boxes
        )

    def _loss_labels_focal(
        self, outputs, targets, indices, matching_costs, num_boxes, log=True
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise ValueError("No logits were found in the outputs")

        source_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["class_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                source_logits.shape[0],
                source_logits.shape[1],
                source_logits.shape[2] + 1,
            ],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                source_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["class_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    @torch.no_grad()
    def loss_uncertainty(self, outputs, targets, indices, matching_costs, num_boxes):
        nonzero_uncertainty_list = []
        for target, index, matching_cost in zip(targets, indices, matching_costs):
            nonzero_index = target["rel"][index[1], :, :][:, index[1], :].nonzero()
            uncertainty = matching_cost.sigmoid()
            nonzero_uncertainty_list.append(
                uncertainty[nonzero_index[:, 0]] * uncertainty[nonzero_index[:, 1]]
            )
        losses = {"uncertainty": torch.cat(nonzero_uncertainty_list).mean()}
        return losses

    def loss_boxes(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        assert "pred_masks" in outputs, "No predicted masks found in outputs"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = nn.functional.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _loss_relations_distillation(
        self,
        pred_fine,  # Teacher Logits (N, N, 50)
        pred_super,  # Student Logits (N, N, 3)
        target_rel,  # GT (N, N, 50)
    ):
        """
        Computes Weighted Cross Entropy + Knowledge Distillation on POSITIVE relations only.
        Discards hard balancing/negative sampling.
        """
        active_indices = target_rel.nonzero(as_tuple=False)

        if active_indices.numel() == 0:
            return {"loss_rel": pred_super.sum() * 0.0}

        i, j = active_indices[:, 0], active_indices[:, 1]

        logits_super_active = pred_super[i, j]  # (K, 3)
        logits_fine_active = pred_fine[i, j]  # (K, 50)

        gt_classes_fine = active_indices[:, 2]  # (K,)
        target_super = self.orig2fam[gt_classes_fine]  # (K,)

        # self.super_loss must be nn.CrossEntropyLoss(weight=..., reduction='none')
        loss_ce = self.super_loss(logits_super_active, target_super).mean()

        # KL Divergence
        with torch.no_grad():
            probs_fine = torch.sigmoid(logits_fine_active)  # (K, 50)

            probs_teacher_super = torch.zeros_like(logits_super_active)
            for fam_id in range(3):
                mask = self.orig2fam == fam_id
                # Sum probabilities of all children in this family
                probs_teacher_super[:, fam_id] = probs_fine[:, mask].sum(dim=1)

            probs_teacher_super = probs_teacher_super / (
                probs_teacher_super.sum(dim=1, keepdim=True) + 1e-6
            )

        # Student Log-Probs
        log_probs_student = F.log_softmax(logits_super_active, dim=1)

        # KL Div: sum(P_teach * (log P_teach - log P_stud))
        # reduction='none' so we can apply adaptive weights
        loss_distill = F.kl_div(
            log_probs_student, probs_teacher_super, reduction="batchmean"
        )

        # --- C. Total Loss ---
        # You can tune alpha (e.g., 1.0)
        alpha = 1.0
        total_loss = loss_ce + (alpha * loss_distill)

        return {
            "loss_rel": total_loss,
            "loss_ce": loss_ce,
            "loss_distill": loss_distill,
        }

    def loss_relations(self, outputs, targets, indices, matching_costs, num_boxes):
        losses = []
        connect_losses = []
        geo_losses = []
        sem_losses = []
        poss_losses = []
        super_losses = []

        for i, ((src_index, target_index), target, matching_cost) in enumerate(
            zip(indices, targets, matching_costs)
        ):
            full_index = torch.arange(self.num_object_queries)
            uniques, counts = torch.cat([full_index, src_index]).unique(
                return_counts=True
            )
            full_src_index = torch.cat([src_index, uniques[counts == 1]])
            full_target_index = torch.cat(
                [target_index, torch.arange(len(target_index), self.num_object_queries)]
            )
            full_matching_cost = torch.cat(
                [
                    matching_cost,
                    torch.full(
                        (self.num_object_queries - len(matching_cost),),
                        self.nonmatching_cost,
                        device=matching_cost.device,
                    ),
                ]
            )

            target_rel = target["rel"][full_target_index][
                :, full_target_index
            ]  # [num_obj_queries, num_obj_queries, config.num_rel_labels]

            rel_index = torch.nonzero(target_rel)
            target_connect = torch.zeros(
                target_rel.shape[0], target_rel.shape[1], 1, device=target_rel.device
            )
            target_connect[rel_index[:, 0], rel_index[:, 1]] = 1

            pred_connectivity = outputs["pred_connectivity"][i, full_src_index][
                :, full_src_index
            ]

            # Connectivity loss
            loss = self.connectivity_loss(pred_connectivity, target_connect)
            connect_losses.append(loss)

            if self.hierarchical and isinstance(outputs["pred_rel"], tuple):
                raw_pred_fine, raw_pred_super = outputs["pred_rel"]

                pred_fine = raw_pred_fine[i, full_src_index][:, full_src_index]
                pred_super = raw_pred_super[i, full_src_index][:, full_src_index]

                loss_distill_dict = self._loss_relations_distillation(
                    pred_fine,
                    pred_super,
                    target_rel,
                )

                # The dictionary contains the summed loss
                losses.append(loss_distill_dict["loss_rel"])

            else:
                pred_rel = outputs["pred_rel"][i, full_src_index][:, full_src_index]
                if self.model_training:
                    loss = self._loss_relations(
                        pred_rel, target_rel, full_matching_cost, 
                        self.rel_sample_negatives, self.rel_sample_nonmatching
                    )
                else:
                    loss = self._loss_relations(pred_rel, target_rel, full_matching_cost, None, None)
                losses.append(loss)

        main_loss_dict = {
            "loss_rel": torch.stack(losses).mean() if losses else torch.tensor(0.0, device=outputs["logits"].device),
            "loss_connectivity": torch.stack(connect_losses).mean() if connect_losses else torch.tensor(0.0, device=outputs["logits"].device),
        }
        return main_loss_dict

    def _select_relation_indices(
        self,
        pred_rel,
        target_rel,
        matching_cost,
        nonmatching_cost,
        rel_sample_negatives,
        rel_sample_nonmatching,
        largest_neg,
        largest_non,
    ):
        """
        Returns:
            relation_indices  – (M, 3) LongTensor with [i,j,k] triplets
            weight            – (M,)   FloatTensor with (1-u_i)(1-u_j)
        """

        device = pred_rel.device
        matched = matching_cost != nonmatching_cost

        N, _, R = pred_rel.shape
        num_obj = int(matched.sum())

        true_idx = (target_rel[:num_obj, :num_obj, :] == 1).nonzero(as_tuple=False)

        false_idx = (target_rel[:num_obj, :num_obj, :] != 1).nonzero(as_tuple=False)

        nonmatch_mask = torch.outer(matched, matched)  # N,N bool
        nonmatch_idx = (
            nonmatch_mask.unsqueeze(-1).expand(N, N, R).nonzero(as_tuple=False)
        )

        num_true = len(true_idx)

        # subsample false / non-match
        def _sample(idx, k_mult, largest):
            if not k_mult or num_true == 0 or idx.numel() == 0:
                return torch.empty(0, 3, dtype=torch.long, device=device)

            k = min(k_mult * num_true, idx.shape[0])

            if k == 0:
                return torch.empty(0, 3, dtype=torch.long, device=device)

            if largest:
                scores = pred_rel[idx[:, 0], idx[:, 1], idx[:, 2]]
                keep = torch.topk(scores, k, largest=True).indices
                return idx[keep]

            # random
            assert 0, "dont reach there yet"
            rand = torch.randperm(idx.shape[0], device=device)[:k]
            return idx[rand]

        false_idx = _sample(false_idx, rel_sample_negatives, largest_neg)
        nonmatch_idx = _sample(nonmatch_idx, rel_sample_nonmatching, largest_non)

        relation_idx = torch.cat([true_idx, false_idx, nonmatch_idx], dim=0)

        # adaptive-smoothing weight
        u = matching_cost.sigmoid()  # (N,)
        weights = (1 - u[relation_idx[:, 0]]) * (1 - u[relation_idx[:, 1]])

        return relation_idx, weights

    def _hierarchical_relation_loss(
        self,
        pred_geo,
        pred_poss,
        pred_sem,  # (N,N,g) (N,N,p) (N,N,s) log-prob
        pred_super,  # (N,N,3)           logits
        target_rel,  # (N,N,R)           one-hot
        matching_cost,  # (N,)              Hungarian cost
        rel_sample_negatives=None,
        rel_sample_nonmatching=None,
    ):
        """
        Returns four scalars (geo, poss, sem, super) **per image**.
        Sampling parameters default to `self.rel_sample_negatives`
        """
        assert pred_geo.size(-1) == self.num_geometric
        assert pred_poss.size(-1) == self.num_possessive
        assert pred_sem.size(-1) == self.num_semantic

        tgt_geo = target_rel.new_zeros(
            (target_rel.size(0), target_rel.size(0), self.num_geometric)
        )
        tgt_poss = target_rel.new_zeros(
            (target_rel.size(0), target_rel.size(0), self.num_possessive)
        )
        tgt_sem = target_rel.new_zeros(
            (target_rel.size(0), target_rel.size(0), self.num_semantic)
        )

        idx = torch.nonzero(target_rel, as_tuple=False)
        if idx.numel() > 0:
            i, j, orig = idx[:, 0], idx[:, 1], idx[:, 2]
            fam = self.orig2fam[orig]
            fam_idx = self.orig2famidx[orig]

            if (fam == 0).any():
                tgt_geo[i[fam == 0], j[fam == 0], fam_idx[fam == 0]] = 1.0
            if (fam == 1).any():
                tgt_poss[i[fam == 1], j[fam == 1], fam_idx[fam == 1]] = 1.0
            if (fam == 2).any():
                tgt_sem[i[fam == 2], j[fam == 2], fam_idx[fam == 2]] = 1.0

        w_obj = 1.0 - matching_cost.sigmoid().detach()
        w_pair = torch.outer(w_obj, w_obj)  # NxN

        # super-relation loss
        mask_rel = target_rel.sum(-1) > 0  # NxN
        ij = mask_rel.nonzero(as_tuple=False)
        assert torch.equal(pred_super[mask_rel], pred_super[ij[:, 0], ij[:, 1]])
        assert torch.equal(target_rel[mask_rel], target_rel[ij[:, 0], ij[:, 1]])

        if mask_rel.any():
            preds = pred_super[mask_rel]

            fine_tgt = target_rel[mask_rel]

            M = preds.shape[0]
            super_tgt = torch.zeros((M, 3), device=preds.device, dtype=torch.float32)

            idx = fine_tgt.nonzero()
            pair_idx = idx[:, 0]
            rel_idx = idx[:, 1]
            fam_idx = self.orig2fam[rel_idx]

            super_tgt.index_put_(
                (pair_idx, fam_idx), torch.tensor(1.0, device=super_tgt.device)
            )
            super_loss_vec = self.super_loss(pred_super[mask_rel], super_tgt)
            super_loss = super_loss_vec.mean()
        else:
            super_loss = (pred_super * 0).sum()

        if self.rel_sample_negatives is None and self.rel_sample_nonmatching is None:

            def _loss(pred, tgt, loss_fn):
                mask = tgt.sum(-1) > 0  # (N,N) bool
                if not mask.any():
                    return (pred * 0).sum()  # keep 0 gradient without detaching

                cls_ind = tgt[mask].argmax(-1)
                loss_vec = loss_fn(pred[mask], cls_ind)
                weight = w_pair[mask]

                return (loss_vec * weight).mean()

            geo_loss = _loss(pred_geo, tgt_geo, self.geo_loss)
            poss_loss = _loss(pred_poss, tgt_poss, self.poss_loss)
            sem_loss = _loss(pred_sem, tgt_sem, self.sem_loss)
            return geo_loss, poss_loss, sem_loss, super_loss

        def _sampled_loss_family(pred, tgt, loss_fn):
            idx, w = self._select_relation_indices(
                pred,
                tgt,
                matching_cost,
                self.nonmatching_cost,
                rel_sample_negatives,
                rel_sample_nonmatching,
                self.rel_sample_negatives_largest,
                self.rel_sample_nonmatching_largest,
            )
            if idx.numel() == 0:
                return (pred * 0).sum()  # keep 0 gradient without detaching

            logp = pred[idx[:, 0], idx[:, 1]]  # (M, C_fam)
            c = tgt[idx[:, 0], idx[:, 1]].argmax(-1)  # (M,)

            return (loss_fn(logp, c) * w).mean()

        geo_loss = _sampled_loss_family(pred_geo, tgt_geo, self.geo_loss)
        poss_loss = _sampled_loss_family(pred_poss, tgt_poss, self.poss_loss)
        sem_loss = _sampled_loss_family(pred_sem, tgt_sem, self.sem_loss)

        return geo_loss, poss_loss, sem_loss, super_loss

    def _loss_relations(
        self,
        pred_rel,
        target_rel,
        matching_cost,
        rel_sample_negatives,
        rel_sample_nonmatching,
    ):
        if (rel_sample_negatives is None) and (rel_sample_nonmatching is None):
            weight = 1.0 - matching_cost.sigmoid()
            weight = torch.outer(weight, weight)
            target_rel = target_rel * weight.unsqueeze(-1)
            loss = self.rel_loss(pred_rel, target_rel).mean(-1).reshape(-1)
        else:
            matched = matching_cost != self.nonmatching_cost
            num_target_objects = sum(matched)

            true_indices = target_rel[
                :num_target_objects, :num_target_objects, :
            ].nonzero()
            false_indices = (
                target_rel[:num_target_objects, :num_target_objects, :] != 1.0
            ).nonzero()
            nonmatching_indices = (
                torch.outer(matched, matched)
                .unsqueeze(-1)
                .repeat(1, 1, self.num_rel_labels)
                != True
            ).nonzero()

            num_target_relations = len(true_indices)
            if rel_sample_negatives is not None:
                if rel_sample_negatives == 0 or num_target_relations == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_negatives_largest:
                        false_sample_scores = pred_rel[
                            false_indices[:, 0],
                            false_indices[:, 1],
                            false_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            false_sample_scores,
                            min(
                                num_target_relations * rel_sample_negatives,
                                false_sample_scores.shape[0],
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(false_indices.size(0)),
                                min(
                                    num_target_relations * rel_sample_negatives,
                                    false_indices.size(0),
                                ),
                            ),
                            device=false_indices.device,
                        )
                false_indices = false_indices[sampled_idx]
            if rel_sample_nonmatching is not None:
                if rel_sample_nonmatching == 0 or num_target_relations == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_nonmatching_largest:
                        nonmatching_sample_scores = pred_rel[
                            nonmatching_indices[:, 0],
                            nonmatching_indices[:, 1],
                            nonmatching_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            nonmatching_sample_scores,
                            min(
                                num_target_relations * rel_sample_nonmatching,
                                nonmatching_indices.size(0),
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(nonmatching_indices.size(0)),
                                min(
                                    num_target_relations * rel_sample_nonmatching,
                                    nonmatching_indices.size(0),
                                ),
                            ),
                            device=nonmatching_indices.device,
                        )
                nonmatching_indices = nonmatching_indices[sampled_idx]

            relation_indices = torch.cat(
                [true_indices, false_indices, nonmatching_indices]
            )
            pred_rel = pred_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]
            target_rel = target_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]

            weight = 1.0 - matching_cost.sigmoid()
            weight = (
                weight[relation_indices[:, 0]] * weight[relation_indices[:, 1]]
            )  # (1-u_sub)(1-u_obj)
            target_rel = target_rel * weight
            loss = self.rel_loss(pred_rel, target_rel)
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, matching_costs, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "relations": self.loss_relations,
            "uncertainty": self.loss_uncertainty,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, matching_costs, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "auxiliary_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, matching_costs = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, matching_costs, num_boxes
                )
            )

        hier_losses = None
        if hasattr(self, "hierarchical_losses") and self.hierarchical:
            hier_losses = self.hierarchical_losses
            delattr(self, "hierarchical_losses")

        if "pred_rels" in outputs:
            for pred_rel in outputs["pred_rels"]:
                outputs["pred_rel"] = pred_rel
                _loss_dict = self.loss_relations(
                    outputs, targets, indices, matching_costs, num_boxes
                )
                losses["loss_rel"] += _loss_dict["loss_rel"]

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):

                indices, matching_costs = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss in ["masks", "relations", "uncertainty"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(
                        loss,
                        auxiliary_outputs,
                        targets,
                        indices,
                        matching_costs,
                        num_boxes,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices, matching_costs = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ["masks", "relations", "uncertainty"]:
                    continue
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, matching_costs, num_boxes
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, hier_losses
