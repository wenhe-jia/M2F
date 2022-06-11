# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

LIP_SEMSEG_CATEGORIES = [
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
    'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
]

# ==== Predefined splits for raw CIHP images ===========
_PREDEFINED_SPLITS = {
    "lip_semseg_train": ("Training/Images/", "Training/Category_ids/", "annotations/LIP_train.json"),
    "lip_semseg_val": ("Validation/Images/", "Validation/Category_ids/", "annotations/LIP_val.json"),
}

LIP_FLIP_MAP = (
    (14, 15),  # ("Left-arm", "Right-arm"),
    (16, 17),  # ("Left-leg", "Right-leg"),
    (18, 19),  # ("Left-shoe", "Right-shoe"),
)

def register_lip_semseg(root):
    root = os.path.join(root, "lip")
    for name, (image_dir, gt_dir, json_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        json_dir = os.path.join(root, json_dir)
        # print(image_dir, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=LIP_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            json_file=json_dir,
            flip_map=LIP_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_lip_semseg(_root)
