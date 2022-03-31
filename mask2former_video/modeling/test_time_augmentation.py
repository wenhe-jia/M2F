# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import sys

import numpy as np
from contextlib import contextmanager
from itertools import count
from typing import List
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.config import configurable
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
    Augmentation,
)
from detectron2.structures import Boxes, Instances

__all__ = ["DatasetMapperTTA_video"]


class DatasetMapperTTA_video:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool, reverse: bool):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip
        self.reverse = reverse

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
            "reverse": cfg.TEST.AUG.REVERSE,
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            tmp_list = []
            resize = ResizeShortestEdge(min_size, self.max_size)
            tmp_list.append(resize)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                tmp_list.append(flip)
                aug_candidates.append([resize, flip])  # resize + flip
            if self.reverse:
                tmp_list.append('reverse')
                if self.flip:
                    aug_candidates.append([resize, 'reverse'])  # resize + reverse
            if self.flip and self.reverse:
                aug_candidates.append(tmp_list)  # resize + reverse + flip

        ret = []
        for aug in aug_candidates:
            isreverse = 'reverse' in aug
            if isreverse:
                aug.remove('reverse')

            for nf, frame in enumerate(dataset_dict["image"]):

                # process the input
                numpy_image = frame.permute(1, 2, 0).numpy()
                shape = numpy_image.shape
                orig_shape = (dataset_dict["height"], dataset_dict["width"])
                if nf == 0:
                    if shape[:2] != orig_shape:
                        # It transforms the "original" image in the dataset to the input image
                        pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
                    else:
                        pre_tfm = NoOpTransform()

                # Apply all the augmentations

                new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
                torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))
                if nf == 0:
                    dic = copy.deepcopy(dataset_dict)
                    dic["transforms"] = pre_tfm + tfms
                    dic["image"] = []

                dic["image"].append(torch_image)
            if isreverse:
                dic["transforms"] += 'reverse'
                dic["image"] = dic["image"][::-1]

            ret.append(dic)
        # print('lllllllll', len(ret))
        # print(len(ret[0]['image']))
        return ret
