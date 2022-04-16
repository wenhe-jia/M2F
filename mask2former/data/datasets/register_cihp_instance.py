# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

CIHP_INSSEG_CATEGORIES = [
    # {'id': 0, 'name': 'background'},
    {'id': 1, 'name': 'Hat'},
    {'id': 2, 'name': 'Hair'}, {'id': 3, 'name': 'Gloves'},
    {'id': 4, 'name': 'Sunglasses'}, {'id': 5, 'name': 'UpperClothes'},
    {'id': 6, 'name': 'Dress'}, {'id': 7, 'name': 'Coat'},
    {'id': 8, 'name': 'Socks'}, {'id': 9, 'name': 'Pants'},
    {'id': 10, 'name': 'Torso-skin'}, {'id': 11, 'name': 'Scarf'},
    {'id': 12, 'name': 'Skirt'}, {'id': 13, 'name': 'Face'},
    {'id': 14, 'name': 'Left-arm'}, {'id': 15, 'name': 'Right-arm'},
    {'id': 16, 'name': 'Left-leg'}, {'id': 17, 'name': 'Right-leg'},
    {'id': 18, 'name': 'Left-shoe'}, {'id': 19, 'name': 'Right-shoe'},
]

CIHP_FLIP_MAP = ((13, 14), (15, 16), (17, 18))


_PREDEFINED_SPLITS = {
    "cihp_part_instance_train":("cihp/Training/Images/", "cihp/annotations/CIHP_part_instance_train.json",),
    "cihp_part_instance_val":("cihp/Validation/Images/", "cihp/annotations/CIHP_part_instance_val.json",),
}


def _get_cihp_instances_meta():
    thing_ids = [k["id"] for k in CIHP_INSSEG_CATEGORIES]
    assert len(thing_ids) == 19, len(thing_ids)
    # Mapping from the incontiguous CIHP category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CIHP_INSSEG_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "parsing_flip_map": CIHP_FLIP_MAP,
    }
    return ret


def register_all_cihp_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_cihp_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cihp_instance(_root)
