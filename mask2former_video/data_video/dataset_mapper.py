# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation, make_coco_transforms

from .image_to_seq_augmenter import ImageToSeqAugmenter

from PIL import Image
from pycocotools import mask as coco_mask

__all__ = ["YTVISDatasetMapper", "YTVISCOCOJointDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)
        
        # print('+ dataset dict: ', type(dataset_dict), dataset_dict.keys())
        # for k, v in dataset_dict.items():
        #     print('++ ', k, ': ', type(v))
        #     if isinstance(v, list):
        #         print('+++ ', k, ' len: ', len(v), type(v[0]))
        #         if k == "image":
        #             print('++++ image: ', v[0].size(), v[0].device)
        #         elif k == 'file_names':
        #             print('++++ file_name: ', v[0])
        #         else:
        #             print('++++ single instance gt_boxes:', v[0].get('gt_boxes'))
        #             print('++++ single instance gt_classes:', v[0].get('gt_classes'))
        #             print('++++ single instance gt_masks:', v[0].get('gt_masks'), v[0].get('gt_masks').tensor, v[0].get('gt_masks').tensor.size())
        #             print('++++ single instance gt_ids:', v[0].get('gt_ids'))
        #     else:
        #         print('+++ ', k, ': ', v)
        return dataset_dict


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks
        #ytvis19
        # self.category_map = {1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 36:33, 41:5, 42:27, 43:40, 74:24}
        # ytvis21, dataset category id map
        # self.category_map = {1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 36:31, 41:29, 42:33, 43:34, 74:24}
        # ytvis, loaded category id map
        self.category_map = {0:25, 1:22, 2:4, 3:22, 4:0, 6:35, 7:36, 8:3, 14:2, 15:5, 16:8, 17:18, 19:6, 20:11, 21:1, 22:39, 23:17, 31:30, 36:28, 37:32, 38:33, 64:23}

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # classes = [obj["category_id"] for obj in anno]  # map coco category id to YTVIS-2019 category id
        classes = [self.category_map[obj["category_id"]] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # target["orig_size"] = torch.as_tensor([int(h), int(w)])
        # target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class YTVISCOCOJointDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        train_scales: tuple = (),
        train_size_max: int = 1333
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes

        # for coco2seq
        self.train_scales = train_scales
        self.train_size_max = train_size_max
        self.coco_augmentations = make_coco_transforms('train' if self.is_train else 'val', self.train_scales)
        self.prepare = ConvertCocoPolysToMask(True)  # default kwargs: return_masks
        self.num_frames = sampling_frame_num
        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-20, 20), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "train_scales": cfg.INPUT.MIN_SIZE_TRAIN,
            "train_size_max": cfg.INPUT.MAX_SIZE_TRAIN
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # print('+ dataset_dict: ', dataset_dict)

        if 'length' not in dataset_dict:  # coco pseudo video processing mode
            # print('load coco data ~~~~~~~~~~~~~')
            img = Image.open(dataset_dict['file_name']).convert('RGB')
            target = {'image_id': dataset_dict['image_id'], 'annotations': dataset_dict['annotations']}

            img, target = self.prepare(img, target)
            seq_images, seq_instance_masks = [img], [target['masks'].numpy()]
            numpy_masks = target['masks'].numpy()

            numinst = len(numpy_masks)
            # print('-- num instance: ', numinst)
            for t in range(self.num_frames - 1):
                im_trafo, instance_masks_trafo = self.augmenter(np.asarray(img), numpy_masks)
                im_trafo = Image.fromarray(np.uint8(im_trafo))
                seq_images.append(im_trafo)
                seq_instance_masks.append(np.stack(instance_masks_trafo, axis=0))
            seq_images, seq_instance_masks = self.apply_random_sequence_shuffle(seq_images, seq_instance_masks)
            output_inst_masks = []
            for inst_i in range(numinst):
                inst_i_mask = []
                for f_i in range(self.num_frames):
                    inst_i_mask.append(seq_instance_masks[f_i][inst_i])
                output_inst_masks.append(np.stack(inst_i_mask, axis=0))

            output_inst_masks = torch.from_numpy(np.stack(output_inst_masks, axis=0))
            target['masks'] = output_inst_masks.flatten(0, 1)
            target['boxes'] = masks_to_boxes(target['masks'])

            if self.coco_augmentations is not None:
                img, target = self.coco_augmentations(seq_images, target, self.num_frames)
            # if len(target['labels']) > 0 and len(target['labels']) <= 25:
            #     pass
            # elif len(target['labels']) == 0:
            #     print('no objects in current coco image, idx: {}, filename: {}'.format(dataset_dict["image_id"], dataset_dict["file_name"]))
            # else:
            #     target['labels'] = target['labels'][:25]

            for inst_id in range(len(target['boxes'])):
                if target['masks'][inst_id].max() < 1:
                    target['boxes'][inst_id] = torch.zeros(4).to(target['boxes'][inst_id])

            target['boxes'] = target['boxes'].clamp(1e-6)
            # print('+++ transformed img and target: ', type(img), type(target))
            # print('+++ img: ', len(img), type(img[0]), img[0].size())
            # print('+++ target: ', target.keys())
            # for k, v in target.items():
            #     if k == 'masks':
            #         print('++++ ', k, ': ', v.size(), '|', v.device, '|', np.unique(v.numpy()))
            #     elif k == 'boxes':
            #         print('++++ ', k, ': ', v.size(), '|', v.device)
            #     else:
            #         print('++++ ', k, ': ', v.size(), '|', v.device, '|', v)

            # prepare single pseudo video dataloader outputs in m2f style
            dataset_dict["length"] = self.num_frames
            dataset_dict["file_names"] = [dataset_dict["file_name"]] * self.num_frames
            del dataset_dict["file_name"]
            dataset_dict["image"] = img

            dataset_dict["instances"] = []
            # print(img[0].size(), '-------')
            aug_h, aug_w = img[0].size()[1:]
            num_ins = int(target["boxes"].size()[0] / self.num_frames)
            boxes = copy.deepcopy(target["boxes"]).reshape(num_ins, self.num_frames, target["boxes"].size()[-1])  # (num_instance, num_frame, 4)
            masks = copy.deepcopy(target["masks"]).reshape(num_ins, self.num_frames, target["masks"].size()[1], target["masks"].size()[2])  # (num_instance, num_frame, H, W)
            for frame_idx in range(self.num_frames):
                # create Instances object for current frame
                instances_per_frame = Instances((aug_h, aug_w))
                # add gt obj ids to Instances
                gt_ids = list(range(num_ins))
                instances_per_frame.gt_ids = torch.tensor(gt_ids)
                # add gt obj classes to Instances
                classes = target["labels"]
                instances_per_frame.gt_classes = classes
                # add gt obj boxes to Instances
                boxes_per_frame = boxes[:, frame_idx, :]  # (num_instance, 4)
                instances_per_frame.gt_boxes = Boxes(boxes_per_frame)
                # add gt obj masks to Instances
                masks_per_frame = masks[:, frame_idx, :, :]  # (num_instance, H, W)
                instances_per_frame.gt_masks = BitMasks(masks_per_frame)

                dataset_dict['instances'].append(instances_per_frame)

        else:  # ytvis processing mode
            # print('load ytvis data ~~~~~~~~~~~~~')

            video_length = dataset_dict["length"]
            if self.is_train:
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)
            else:
                selected_idx = range(video_length)

            video_annos = dataset_dict.pop("annotations", None)
            file_names = dataset_dict.pop("file_names", None)

            if self.is_train:
                _ids = set()
                for frame_idx in selected_idx:
                    _ids.update([anno["id"] for anno in video_annos[frame_idx]])
                ids = dict()
                for i, _id in enumerate(_ids):
                    ids[_id] = i

            dataset_dict["image"] = []
            dataset_dict["instances"] = []
            dataset_dict["file_names"] = []
            for frame_idx in selected_idx:
                dataset_dict["file_names"].append(file_names[frame_idx])

                # Read image
                image = utils.read_image(file_names[frame_idx], format=self.image_format)
                utils.check_image_size(dataset_dict, image)

                aug_input = T.AugInput(image)
                transforms = self.augmentations(aug_input)
                image = aug_input.image

                image_shape = image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

                if (video_annos is None) or (not self.is_train):
                    continue

                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]
                sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

                for _anno in annos:
                    idx = ids[_anno["id"]]
                    sorted_annos[idx] = _anno
                _gt_ids = [_anno["id"] for _anno in sorted_annos]

                instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
                instances.gt_ids = torch.tensor(_gt_ids)
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances(instances)
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
                dataset_dict["instances"].append(instances)

        return dataset_dict

    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks

class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict
