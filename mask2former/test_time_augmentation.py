# Copyright (c) Facebook, Inc. and its affiliates.
import copy, cv2
import logging
from itertools import count

import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA

from detectron2.data.transforms import (
    RandomFlip,
    apply_augmentations,
)
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from .data.parsing_utils import get_parsing_flip_map, center_to_target_size

__all__ = [
    "ParsingSemanticSegmentorWithTTA",
    "SingleParsingDatasetMapperTTA",
    "SemanticSegmentorWithTTA",
]

class SingleParsingDatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, flip: bool):
        """
        Args:
            flip: whether to apply flipping augmentation
        """

        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "flip": cfg.TEST.AUG.FLIP,
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
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()

        # Create all combinations of augmentations to use
        aug_candidates = [[]]  # each element is a list[Augmentation]
        if self.flip:
            flip = RandomFlip(prob=1.0)
            aug_candidates.append([flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret

# class SingleParsingDatasetMapperTTA:
#     """
#     Implement test-time augmentation for detection data.
#     It is a callable which takes a dataset dict from a detection dataset,
#     and returns a list of dataset dicts where the images
#     are augmented from the input image by the transformations defined in the config.
#     This is used for test-time augmentation.
#     """
#
#     @configurable
#     def __init__(self, test_scales, flip: bool):
#         """
#         Args:
#             min_sizes: list of short-edge size to resize the image to
#             max_size: maximum height or width of resized images
#             flip: whether to apply flipping augmentation
#         """
#         self.test_scales = test_scales
#         self.flip = flip
#
#     @classmethod
#     def from_config(cls, cfg):
#         return {
#             "test_scales": cfg.INPUT.SINGLE_PARSING.SCALES,
#             "flip": cfg.TEST.AUG.FLIP,
#         }
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dict: a dict in standard model input format. See tutorials for details.
#
#         Returns:
#             list[dict]:
#                 a list of dicts, which contain augmented version of the input image.
#                 The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
#                 Each dict has field "transforms" which is a TransformList,
#                 containing the transforms that are used to generate this image.
#         """
#         numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
#         sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
#
#         shape = numpy_image.shape
#         orig_shape = (dataset_dict["height"], dataset_dict["width"])
#         if shape[:2] != orig_shape:
#             # It transforms the "original" image in the dataset to the input image
#             pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
#         else:
#             pre_tfm = NoOpTransform()
#
#         # Create all combinations of augmentations to use
#         aug_candidates = []
#         for test_scale in self.test_scales:
#             aug_candidates.append({"scale": test_scale})
#             if self.flip:
#                 aug_candidates.append(
#                     {
#                         "scale": test_scale,
#                         "flip": RandomFlip(prob=1.0)
#                     }
#                 )
#
#         # Apply all the augmentations
#         ret = []
#         for aug in aug_candidates:
#             tfms = {"scale": aug["scale"]}
#             if "flip" in aug:
#                 new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
#                 tfms["flip"] = True
#             else:
#                 new_image = numpy_image
#                 tfms["flip"] = False
#
#             new_image, new_sem_seg_gt = center_to_target_size(new_image, sem_seg_gt, aug["scale"])
#             torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))
#             torch_gt = torch.as_tensor(new_sem_seg_gt.astype("long"))
#
#             dic = copy.deepcopy(dataset_dict)
#             dic["transforms"] = tfms
#             dic["image"] = torch_image
#             dic["sem_seg"] = torch_gt
#             ret.append(dic)
#         return ret

class ParsingSemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.flip_map = get_parsing_flip_map(self.cfg.DATASETS.TEST)
        self.model = model
        self.insseg_to_semseg = self.cfg.MODEL.MASK_FORMER.TEST.INSSEG_TO_SEMSEG

        if tta_mapper is None:
            if cfg.MODEL.MASK_FORMER.MULTI_PERSON_PARSING:
                tta_mapper = DatasetMapperTTA(cfg)
            else:
                print("\n\n===========\nUsing SingleParsingDatasetMapperTTA\n===========\n\n")
                tta_mapper = SingleParsingDatasetMapperTTA(cfg)
        # if tta_mapper is None:
        #     tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        image_name = input['file_name'].split('/')[-1].split('.')[0]

        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0

        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        flipped_predictions = self.model([input])[0].pop("sem_seg")
                        # if self.insseg_to_semseg:
                        final_predictions = self.flip_parsing_back(flipped_predictions)
                        # else:
                        #     final_predictions = flipped_predictions.flip(dims=[2])
                    else:
                        # cv2.imwrite('/home/user/Program/vis/m2f-cihp/Mask2Former/im.jpg', input['image'].cpu().numpy().transpose(1,2,0))
                        final_predictions = self.model([input])[0].pop("sem_seg")
                        # cv2.imwrite('/home/user/Program/vis/m2f-cihp/Mask2Former/org_pred_{}.png'.format(image_name), final_predictions.argmax(dim=0).cpu().numpy()* 15)
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        # cv2.imwrite('/home/user/Program/vis/m2f-cihp/Mask2Former/im_flip.jpg', input['image'].cpu().numpy().transpose(1,2,0))
                        flipped_predictions = self.model([input])[0].pop("sem_seg")
                        # if self.insseg_to_semseg:
                        final_predictions += self.flip_parsing_back(flipped_predictions)
                        # else:
                        #     final_predictions += flipped_predictions.flip(dims=[2])
                    else:
                        final_predictions += self.model([input])[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions

        return {"sem_seg": final_predictions}


    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def flip_parsing_back(self, predictions):
        spatial_flipback_predictions = predictions.flip(dims=[2])  # (num_cls, H, W)
        spatial_channel_flipback_predictions = copy.deepcopy(spatial_flipback_predictions)

        # channel transaction to flip human part label
        for ori_label, new_label in self.flip_map:
            if self.insseg_to_semseg:
                ori_label += 1
                new_label += 1

            org_channel = spatial_flipback_predictions[ori_label, :, :]
            new_channel = spatial_flipback_predictions[new_label, :, :]

            spatial_channel_flipback_predictions[new_label, :, :] = org_channel
            spatial_channel_flipback_predictions[ori_label, :, :] = new_channel

        return spatial_channel_flipback_predictions


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions = self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions = self.model([input])[0].pop("sem_seg")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions += self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions += self.model([input])[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms
