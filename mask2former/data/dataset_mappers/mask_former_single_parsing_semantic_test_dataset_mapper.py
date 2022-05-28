# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import random, cv2, os
from PIL import Image

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from ..transforms.augmentation_impl import ResizeByAspectRatio, ResizeByScale, RandomCenterRotation
from ..parsing_utils import flip_parsing_semantic_category, center_to_target_size, affine_to_target_size

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["MaskFormerSingleParsingSemanticTestDatasetMapper"]


def build_augmentation(cfg):
    test_size = cfg.INPUT.SINGLE_PARSING.SCALES[0]
    aspect_ratio = test_size[0] * 1.0 / test_size[1]
    # augs = [ResizeByAspectRatio(aspect_ratio, interp=Image.NEAREST)]
    # return augs
    return []


class MaskFormerSingleParsingSemanticTestDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train,
        *,
        augmentations,
        image_format,
        ignore_label,
        test_size,
        size_divisibility,
        parsing_flip_map,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """

        # fmt: off
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.test_size = test_size
        self.size_divisibility = size_divisibility
        self.parsing_flip_map = parsing_flip_map

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[MaskFormerSingleParsingTestDatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=False):
        augs = build_augmentation(cfg)

        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": meta.ignore_label,
            "test_size": cfg.INPUT.SINGLE_PARSING.SCALES[0],
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,  # -1
            "parsing_flip_map": meta.flip_map
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert not self.is_train, "MaskFormerSingleParsingTestDatasetMapper should only be used for testing!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image, sem_seg_gt = center_to_target_size(image, sem_seg_gt, self.test_size)
        # image, sem_seg_gt = affine_to_target_size(image, sem_seg_gt, self.test_size)

        # image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict
