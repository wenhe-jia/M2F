import logging
import numpy as np
from typing import List, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog


__all__ = [
    "get_parsing_flip_map",
    "transform_parsing_insseg_instance_annotations",
]


def get_parsing_flip_map(dataset_names):
    meta = MetadataCatalog.get(dataset_names[0])
    return meta.parsing_flip_map

def transform_parsing_insseg_instance_annotations(
    annotation, transforms, image_size, *, parsing_flip_map=None
):
    """
    Apply transforms to box and segmentation of a single human part instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        parsing_flip_map (tuple(int, int)): hflip label map.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]

        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]

            # change part label if do h_flip
            annotation["category_id"] = flip_cihp_parsing_category(
                annotation["category_id"], transforms, parsing_flip_map
            )
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask

            # change part label if do h_flip
            annotation["category_id"] = flip_cihp_parsing_category(
                annotation["category_id"], transforms, cihp_flip_map
            )
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation


def flip_cihp_parsing_category(category, transforms, flip_map):
    # flip_map: ((13, 14), (15, 16), (17, 18))

    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1  # bool

    if do_hflip:
        for ori_label, new_label in flip_map:
            if category == ori_label:
                category = new_label
            elif category == new_label:
                category = ori_label
    return category
