# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import copy, cv2
import numpy as np
from .data.parsing_utils import compute_parsing_IoP
from .modeling.postprocessing import single_parsing_sem_seg_postprocess


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # evaluate parsing semseg metrics(mIoU) with insseg model
        parsing_on: bool,
        instance_to_semantic: bool,
        parsing_with_human: bool,
        parsing_ins_score_thr: float,
        iop_thresh: float,
        multi_person_parsing: bool
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        # evaluate parsing
        self.parsing_on = parsing_on
        self.instance_to_semantic = instance_to_semantic
        self.parsing_with_human = parsing_with_human
        self.parsing_ins_score_thr = parsing_ins_score_thr
        self.iop_thresh = iop_thresh
        self.multi_person_parsing = multi_person_parsing

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        if "cihp" in cfg.DATASETS.TEST[0]:
            parsing_on = True
            multi_person_parsing = True
        elif "lip" in cfg.DATASETS.TEST[0]:
            parsing_on = True
            multi_person_parsing = False
        else:
            parsing_on = False
            multi_person_parsing = True

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # parsing
            "parsing_on": parsing_on,
            "instance_to_semantic": cfg.MODEL.MASK_FORMER.TEST.PARSING.INSTANCE_TO_SEMANTIC,
            "parsing_with_human": cfg.MODEL.MASK_FORMER.TEST.PARSING.PARSING_WITH_HUMAN,
            "parsing_ins_score_thr": cfg.MODEL.MASK_FORMER.TEST.PARSING.PARSING_INS_SCORE_THR,
            "iop_thresh": cfg.MODEL.MASK_FORMER.TEST.PARSING.IOP_THR,
            "multi_person_parsing": multi_person_parsing,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]  # (B, Q, C+1)
            mask_pred_results = outputs["pred_masks"]  # (B, Q, H, W)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({}) # for each image

                if self.sem_seg_postprocess_before_inference:
                    """
                    TOD: maybe add single parsing postprocess
                    """
                    if not self.parsing_on:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                    elif self.parsing_on and self.multi_person_parsing:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                    elif self.parsing_on and not self.multi_person_parsing:
                        mask_pred_result = retry_if_cuda_oom(single_parsing_sem_seg_postprocess)(
                            mask_pred_result, image_size, input_per_image["crop_box"], height, width
                        )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)  # change device as mask_pred_result

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        if not self.parsing_on:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height,width)
                        elif self.parsing_on and self.multi_person_parsing:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        elif self.parsing_on and not self.multi_person_parsing:
                            r = retry_if_cuda_oom(single_parsing_sem_seg_postprocess)(
                                r, image_size, input_per_image["crop_box"], height, width
                            )
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    if not self.parsing_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r  # compiled in "Instances" structure
                    if self.parsing_on and self.instance_to_semantic:
                        semantic_r = retry_if_cuda_oom(self.semantic_inference_for_instance_model)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["sem_seg"] = semantic_r
                    elif self.parsing_on and not self.insseg_to_semseg and self.parsing_with_human:
                        parsing_r = retry_if_cuda_oom(self.parsing_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["parsing"] = parsing_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # discard non-sense category
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_cls(Q, C)
        # mask_pred(Q, H, W) is already processed to have the same shape as original input by func "sem_seg_postprocess"
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # discard non-sense category
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]  # (topk,), scores_per_image in same shape

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]  # (topk, H_org, W_org)

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()  # (topk, H_org, W_org), binary(float)
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))  # (topk, 4)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # perform SOLO rescore
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)  # pixel score

        # perform QANet pixel score
        # mask_pred = self.restrict_mask_to_fg(mask_pred)
        # im_masks_tl = mask_pred.sigmoid() >= 0.2
        # im_masks_th = mask_pred.sigmoid() >= (1 - 0.2)
        # mask_scores_per_image = (torch.sum(im_masks_th, dim=(1, 2)).to(dtype=torch.float32)
        #                      / torch.sum(im_masks_tl, dim=(1, 2)).to(dtype=torch.float32).clamp(min=1e-6))


        # result.scores = scores_per_image
        # result.scores = scores_per_image * mask_scores_per_image
        result.scores = torch.pow(torch.pow(scores_per_image, 1.0) * torch.pow(mask_scores_per_image, 3.0), 1./1+3)

        result.pred_classes = labels_per_image
        return result

    def semantic_inference_for_instance_model(self, mask_cls, mask_pred):
        # mask_cls(Q, C)
        # mask_pred(Q, H, W) is already processed to have the same shape as original input by func "sem_seg_postprocess"

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # discard non-sense category
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]  # (topk,), scores_per_image in same shape

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]  # (topk, H_org, W_org), mask of a query could be selected more than one time

        # for rescore by pixel score
        binary_pred_masks = (mask_pred > 0).float()  # (topk, H_org, W_org), binary(float), before sigmoid
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * binary_pred_masks.flatten(1)).sum(1) / \
                                (binary_pred_masks.flatten(1).sum(1) + 1e-6)  # pixel score

        pred_scores = scores_per_image * mask_scores_per_image
        pred_labels = labels_per_image
        pred_masks = mask_pred

        '''
        Paste sigmoid results for each category
        '''
        im_h, im_w = mask_pred.shape[-2:]
        semseg_im = [torch.zeros((im_h, im_w), dtype=torch.float32, device=pred_masks.device) + 1e-6]
        for cls_ind in range(self.sem_seg_head.num_classes):  # 19 for CIHP, without background
            keep_ind = torch.where(pred_labels == cls_ind)
            scores_cate = pred_scores[keep_ind]
            masks_cate = pred_masks[keep_ind].sigmoid()

            # paste_time = 0
            semseg_cate = torch.zeros((im_h, im_w), dtype=torch.float32, device=pred_masks.device)
            _indx = scores_cate.argsort()
            for k in range(len(_indx)):
                if scores_cate[_indx[k]] < self.parsing_ins_score_thr:
                    continue
                _ins_mask = masks_cate[_indx[k]] * scores_cate[_indx[k]]
                semseg_cate = torch.where(_ins_mask > 0.5, _ins_mask + semseg_cate, semseg_cate)
                # paste_time += 1

            # if paste_time > 0:
            #     semseg_im.append(semseg_cate / paste_time)
            # else:
            #     semseg_im.append(semseg_cate)

            semseg_im.append(semseg_cate)

        return torch.stack(semseg_im, dim=0)  # (num_cls_ins, H_org, W_org)

        # '''
        # paste label
        # '''
        # semseg_im = torch.zeros((im_h, im_w), dtype=torch.uint8, device=pred_masks.device)
        #
        # _indx = pred_scores.argsort()
        # for k in range(len(_indx)):
        #     if pred_scores[_indx[k]] < self.ins2sem_score_thresh:
        #         continue
        #     _ins_mask = pred_masks[_indx[k]]
        #     _ins_label = pred_labels[_indx[k]]
        #     label_mask = torch.ones(_ins_mask.shape, dtype=torch.uint8, device=pred_masks.device) * (_ins_label + 1)
        #     semseg_im = torch.where(_ins_mask > 0, label_mask, semseg_im)
        #
        # return semseg_im
    def paste_instance_to_semseg_label_map(self, labels, scores, prob_masks):
        '''
        Paste sigmoid results for each category
        '''
        im_h, im_w = prob_masks.shape[-2:]
        semseg_im = [torch.zeros((im_h, im_w), dtype=torch.float32, device=prob_masks.device) + 1e-6]
        for cls_ind in range(self.sem_seg_head.num_classes - 1):  # 0~18 for CIHP, without background
            keep_ind = torch.where(labels == cls_ind + 1)
            scores_cate = scores[keep_ind]
            masks_cate = prob_masks[keep_ind].sigmoid()

            # paste_time = 0
            semseg_cate = torch.zeros((im_h, im_w), dtype=torch.float32, device=prob_masks.device)
            _indx = scores_cate.argsort()
            for k in range(len(_indx)):
                if scores_cate[_indx[k]] < self.parsing_ins_score_thr:
                    continue
                _ins_mask = masks_cate[_indx[k]] * scores_cate[_indx[k]]
                semseg_cate = torch.where(_ins_mask > 0.5, _ins_mask + semseg_cate, semseg_cate)
                # paste_time += 1

            # if paste_time > 0:
            #     semseg_im.append(semseg_cate / paste_time)
            # else:
            #     semseg_im.append(semseg_cate)

            semseg_im.append(semseg_cate)

        # semseg_mask = torch.stack(semseg_im, dim=0)  #
        semseg_mask = torch.stack(semseg_im, dim=0).argmax(dim=0).cpu().numpy()
        return semseg_mask

    def parsing_inference(self, mask_cls, mask_pred):
        # mask_cls(Q, C)
        # mask_pred(Q, H, W) is already processed to have the same shape as original input by func "sem_seg_postprocess"

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # discard non-sense category
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]  # (topk,), scores_per_image in same shape

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]  # (topk, H_org, W_org), mask of a query could be selected more than one time

        # for rescore by pixel score
        binary_pred_masks = (mask_pred > 0).float()  # (topk, H_org, W_org), binary(float), before sigmoid
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * binary_pred_masks.flatten(1)).sum(1) / \
                                (binary_pred_masks.flatten(1).sum(1) + 1e-6)  # pixel score

        pred_scores = scores_per_image * mask_scores_per_image
        pred_labels = labels_per_image
        pred_masks = mask_pred

        # get person instances and part instances
        part_labels = pred_labels[torch.where(pred_labels != 0)[0]]
        part_scores = pred_scores[torch.where(pred_labels != 0)[0]]
        part_masks  = pred_masks[torch.where(pred_labels != 0)[0], :, :]

        person_labels = pred_labels[torch.where(pred_labels == 0)[0]]
        person_scores = pred_scores[torch.where(pred_labels == 0)[0]]
        person_masks = pred_masks[torch.where(pred_labels == 0)[0], :, :]

        person_keep_ind = torch.where(person_scores > 0.)[0]  # self.parsing_ins_score_thr
        person_scores = person_scores[person_keep_ind]
        person_masks = person_masks[person_keep_ind, :, :]

        semseg_res = self.paste_instance_to_semseg_label_map(part_labels, part_scores, part_masks)


        part_res = []
        for part_idx in range(part_labels.shape[0]):
            if part_scores[part_idx] < 0.1:
                continue
            part_res.append(
                {
                    "category_id": part_labels[part_idx].cpu(),
                    "score": part_scores[part_idx].cpu(),
                    "mask": (part_masks[part_idx] > 0.).cpu().numpy().astype(np.uint8),
                }
            )

        human_res = []
        for person_idx in range(person_scores.shape[0]):
            # if person_scores[person_idx] < 0.1:
            #     continue
            human_res.append(
                {
                    "category_id": person_labels[person_idx].cpu(),
                    "score": person_scores[person_idx].cpu(),
                    "mask": (person_masks[person_idx] > 0.).cpu().numpy().astype(np.uint8),
                }
            )

        pars_res = []
        # prepare matching infos, matching is based on
        matching_mtx = torch.zeros((person_scores.shape[0], part_scores.shape[0]), dtype=torch.uint8)
        person_ids = torch.arange(person_masks.shape[0])
        part_ids = torch.arange(part_masks.shape[0])
        for person_id, person_score, person_mask in zip(person_ids, person_scores, person_masks):
            for part_id, part_label, part_score, part_mask in zip(part_ids, part_labels, part_scores, part_masks):
                if part_score > self.parsing_ins_score_thr:
                    iop = compute_parsing_IoP(copy.deepcopy(person_mask > 0), copy.deepcopy(part_mask > 0))

                    if iop > self.iop_thresh:
                        matching_mtx[person_id, part_id] = 1

            matched_part_ids = matching_mtx[person_id]

            parsing_person, parts_pix_score = self.get_person_parsing(
                person_mask,
                part_scores[matched_part_ids],
                part_labels[matched_part_ids],
                part_masks[matched_part_ids]
            )

            pars_res.append(
                {
                    "category_id": 1,
                    "parsing": parsing_person.cpu().numpy(),  # (H, W)
                    "instance_score": person_score.cpu(),
                    "parsing_bbox_score": person_score.cpu(),
                    "part_pixel_scores": parts_pix_score,
                }
            )

        return {
            "semseg_outputs": semseg_res,
            "parsing_outputs": pars_res,
            "part_outputs": part_res,
            "human_outputs": human_res
        }

    def get_person_parsing(self, person_mask, part_scores, part_labels, part_masks):
        im_h, im_w = part_masks.shape[-2:]
        # person_parsing = [torch.zeros((im_h, im_w), dtype=torch.float32, device=part_masks.device) + 1e-6]
        person_parsing = [1 - person_mask.sigmoid()]
        part_pix_scores = []
        for cls_ind in range(1, self.sem_seg_head.num_classes):  # skip class 'person'
            keep_ind = torch.where(part_labels == cls_ind)[0]
            if len(keep_ind) == 0:
                semseg_cate = torch.zeros((im_h, im_w), dtype=torch.float32, device=part_masks.device)
            elif len(keep_ind) == 1:
                semseg_cate = part_masks[keep_ind[0]].sigmoid()
            else:
                scores_cate = part_scores[keep_ind]
                masks_cate = part_masks[keep_ind].sigmoid()

                # keep only one part instance for one person
                # max_idx = scores_cate.argmax()
                # semseg_cate = masks_cate[max_idx, :, :]

                paste_time = 0
                semseg_cate = torch.zeros((im_h, im_w), dtype=torch.float32, device=part_masks.device)
                for part_ind in range(len(keep_ind)):
                    if part_scores[part_ind] < 0.:
                        continue
                    paste_time += 1
                    part_mask = masks_cate[part_ind]
                    semseg_cate = torch.where(part_mask > 0.5, part_mask + semseg_cate, semseg_cate)

                if paste_time > 0:
                    semseg_cate /= paste_time

                # paste_map = torch.ones((im_h, im_w), dtype=torch.float32, device=part_masks.device)
                # semseg_cate = torch.zeros((im_h, im_w), dtype=torch.float32, device=part_masks.device)
                # for part_ind in range(len(keep_ind)):
                #     part_mask = masks_cate[part_ind]
                #
                #     paste_map  = torch.where(part_mask > 0.5, paste_map + 1, paste_map)
                #     semseg_cate = torch.where(part_mask > 0.5, part_mask + semseg_cate, semseg_cate)
                #     semseg_cate /= paste_map

            person_parsing.append(semseg_cate)
            # part pixel score
            part_pix_scores.append(self.pixel_score(semseg_cate).cpu())

        parsing_probs = torch.stack(person_parsing, dim=0)  # (C, H, W)
        parsings = parsing_probs.argmax(dim=0).to(dtype=torch.uint8)

        return parsings, part_pix_scores

    def pixel_score(self, pred, thr=0.5):
        # pred: (H, W)
        # high confidence mask (hcm)
        inst_hcm = (pred >= thr).to(dtype=torch.bool)
        # high confidence value (hcv)
        inst_hcv = torch.sum(pred * inst_hcm, dim=[0, 1]).to(dtype=torch.float32)
        inst_hcm_num = torch.clamp(torch.sum(inst_hcm, dim=[0, 1]).to(dtype=torch.float32), min=1e-6)

        pix_score = inst_hcv / inst_hcm_num

        return pix_score

    def restrict_mask_to_fg(self, pred_mask):
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        assert(len(pred_mask.shape) == 3)

        _pred_mask = copy.deepcopy(pred_mask)
        binary_logits = (_pred_mask > 0).int()

        boxes = torch.zeros(binary_logits.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(binary_logits, dim=1)
        y_any = torch.any(binary_logits, dim=2)
        for idx in range(binary_logits.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )

        _mask = torch.zeros(
            boxes.shape[0], binary_logits.shape[1], binary_logits.shape[2], dtype=torch.float32, device=pred_mask.device
        )
        for ins_id in range(boxes.shape[0]):
            box = boxes[ins_id]
            _mask[ins_id, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

        return pred_mask * _mask



