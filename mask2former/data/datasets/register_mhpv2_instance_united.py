# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

MHPv2_UNITED_INSSEG_CATEGORIES = [
    {'id': 0,  'name': 'Person'},
    {'id': 1, 'name': 'Cap/Hat'},                    {'id': 2,  'name': 'Helmet'},
    {'id': 3, 'name': 'Face'},                       {'id': 4,  'name': 'Hair'},
    {'id': 5, 'name': 'Left-arm'},                   {'id': 6,  'name': 'Right-arm'},
    {'id': 7, 'name': 'Left-hand'},                  {'id': 8,  'name': 'Right-hand'},
    {'id': 9, 'name': 'Protector'},                  {'id': 10, 'name': 'Bikini/bra'},
    {'id': 11, 'name': 'Jacket/Windbreaker/Hoodie'}, {'id': 12, 'name': 'T-shirt'},
    {'id': 13, 'name': 'Polo-shirt'},                {'id': 14, 'name': 'Sweater'},
    {'id': 15, 'name': 'Singlet'},                   {'id': 16, 'name': 'Torso-skin'},
    {'id': 17, 'name': 'Pants'},                     {'id': 18, 'name': 'Shorts/Swim-shorts'},
    {'id': 19, 'name': 'Skirt'},                     {'id': 20, 'name': 'Stockings'},
    {'id': 21, 'name': 'Socks'},
    {'id': 22, 'name': 'Left-boot'},                 {'id': 23, 'name': 'Right-boot'},
    {'id': 24, 'name': 'Left-shoe'},                 {'id': 25, 'name': 'Right-shoe'},
    {'id': 26, 'name': 'Left-highheel'},             {'id': 27, 'name': 'Right-highheel'},
    {'id': 28, 'name': 'Left-sandal'},               {'id': 29, 'name': 'Right-sandal'},
    {'id': 30, 'name': 'Left-leg'},                  {'id': 31, 'name': 'Right-leg'},
    {'id': 32, 'name': 'Left-foot'},                 {'id': 33, 'name': 'Right-foot'},
    {'id': 34, 'name': 'Coat'},                      {'id': 35, 'name': 'Dress'},
    {'id': 36, 'name': 'Robe'},                      {'id': 37, 'name': 'Jumpsuits'},
    {'id': 38, 'name': 'Other-full-body-clothes'},   {'id': 39, 'name': 'Headwear'},
    {'id': 40, 'name': 'Backpack'},                  {'id': 41, 'name': 'Ball'},
    {'id': 42, 'name': 'Bats'},                      {'id': 43, 'name': 'Belt'},
    {'id': 44, 'name': 'Bottle'},                    {'id': 45, 'name': 'Carrybag'},
    {'id': 46, 'name': 'Cases'},                     {'id': 47, 'name': 'Sunglasses'},
    {'id': 48, 'name': 'Eyewear'},                   {'id': 49, 'name': 'Gloves'},
    {'id': 50, 'name': 'Scarf'},                     {'id': 51, 'name': 'Umbrella'},
    {'id': 52, 'name': 'Wallet/Purse'},              {'id': 53, 'name': 'Watch'},
    {'id': 54, 'name': 'Wristband'},                 {'id': 55, 'name': 'Tie'},
    {'id': 56, 'name': 'Other-accessaries'},         {'id': 57, 'name': 'Other-upper-body-clothes'},
    {'id': 58, 'name': 'Other-lower-body-clothes'},
]
MHPv2_UNITED_FLIP_MAP = ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33))

_PREDEFINED_SPLITS = {
    "mhpv2_united_instance_train":("mhpv2/Training/Images/", "mhpv2/annotations/MHP-v2_united_instance_train.json",),
    "mhpv2_united_instance_val":("mhpv2/Validation/Images/", "mhpv2/annotations/MHP-v2_united_instance_val.json",),
}


def _get_mhpv2_united_instances_meta():
    thing_ids = [k["id"] for k in MHPv2_UNITED_INSSEG_CATEGORIES]
    assert len(thing_ids) == 59, len(thing_ids)
    # Mapping from the incontiguous CIHP category id to an id in (0, 99)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in MHPv2_UNITED_INSSEG_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "flip_map": MHPv2_UNITED_FLIP_MAP,
    }
    return ret


def register_all_mhpv2_united_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_mhpv2_united_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mhpv2_united_instance(_root)
