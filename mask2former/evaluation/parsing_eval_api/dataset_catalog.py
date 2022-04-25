import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'
_ANN_TYPES = 'annotation_types'
_ANN_FIELDS = 'annotation_fields'

# TODO: coco_panoptic, coco_densepose, cocohumanparts, voc, cityscape, object365v2, MHP, VIP, LaPa, ATR, PPP, VSPW, MSL
COMMON_DATASETS = {
    'cifar10': {
        _IM_DIR: _DATA_DIR + '/cifar',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls',),
        _ANN_FIELDS: {
            'num_images': 60000,    # train: 50000, val: 10000
            'cls': {
                'num_classes': 10,
            },
        },
    },
    'cifar100': {
        _IM_DIR: _DATA_DIR + '/cifar',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls',),
        _ANN_FIELDS: {
            'num_images': 60000,    # train: 50000, val: 10000
            'cls': {
                'num_classes': 100,
            },
        },
    },
    'imagenet1k_2017_train': {
        _IM_DIR: _DATA_DIR + '/ILSVRC2017/Data/CLS-LOC/train',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls', ),
        _ANN_FIELDS: {
            'num_images': 1281167,
            'cls': {
                'num_classes': 1000,
            },
        },
    },
    'imagenet1k_2017_val': {
        _IM_DIR: _DATA_DIR + '/ILSVRC2017/Data/CLS-LOC/val',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls',),
        _ANN_FIELDS: {
            'num_images': 50000,
        },
    },
    'imagenet21k_train': {
        _IM_DIR: _DATA_DIR + '/ImageNet21K/Data/CLS-LOC/train',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls', ),
        _ANN_FIELDS: {
            'num_images': 14197122,
            'cls': {
                'num_classes': 21841,
            },
        },
    },
    'imagenet21k_val': {
        _IM_DIR: _DATA_DIR + '/ImageNet21K/Data/CLS-LOC/val',
        _ANN_FN: _DATA_DIR,
        _ANN_TYPES: ('cls',),
        _ANN_FIELDS: {
            'num_images': -1,
        },
    },
    'lvis_v0.5_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v0.5_val_2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1230,
                'num_instances': 50763,
            },
            'mask': {
                'num_classes': 1230,
                'num_instances': 50763,
            },
        },
    },
    'lvis_v0.5_test': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v0.5_image_info_test.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 19761,
            'bbox': {
                'num_classes': 1230,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1230,
                'num_instances': -1,  # no annotations
            },
        },
    },
    'lvis_v1_train': {
        _IM_DIR: _DATA_DIR + '/COCO',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_train.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 100170,
            'bbox': {
                'num_classes': 1023,
                'num_instances': 1270141,
            },
            'mask': {
                'num_classes': 1023,
                'num_instances': 1270141,
            },
            'semseg': {
                'num_classes': 1024,
                'num_instances': 1270141,
                'seg_json': _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_train.json',
                # 'seg_root': _DATA_DIR,
                'semseg_format': 'poly',
            },
        },
    },
    'lvis_v1_train_fixed': {    # fixed the inaccurate bboxes according to semseg annotations
        _IM_DIR: _DATA_DIR + '/COCO',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_train_fixed.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 100170,
            'bbox': {
                'num_classes': 1023,
                'num_instances': 1270141,
            },
            'mask': {
                'num_classes': 1023,
                'num_instances': 1270141,
            },
            'semseg': {
                'num_classes': 1024,
                'num_instances': 1270141,
                'seg_json': _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_train_fixed.json',
                # 'seg_root': _DATA_DIR,
                'semseg_format': 'poly',
            },
        },
    },
    'lvis_v1_val': {
        _IM_DIR: _DATA_DIR + '/COCO',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 19809,
            'bbox': {
                'num_classes': 1023,
                'num_instances': 244707,
            },
            'mask': {
                'num_classes': 1023,
                'num_instances': 244707,
            },
            'semseg': {
                'num_classes': 1024,
                'num_instances': 244707,
                'seg_json': _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_val.json',
                # 'seg_root': _DATA_DIR,
                'semseg_format': 'poly',
            },
        },
    },
    'lvis_v1_test-dev': {
        _IM_DIR: _DATA_DIR + '/COCO',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_image_info_test_dev.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 19822,
            'bbox': {
                'num_classes': 1023,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1023,
                'num_instances': -1,  # no annotations
            },
            'semseg': {
                'num_classes': 1024,
                'num_instances': -1,  # no annotations
            },
        },
    },
    'lvis_v1_test-challenge': {
        _IM_DIR: _DATA_DIR + '/COCO',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/lvis/lvis_v1_image_info_test_challenge.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 19822,
            'bbox': {
                'num_classes': 1023,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1023,
                'num_instances': -1,  # no annotations
            },
            'semseg': {
                'num_classes': 1024,
                'num_instances': -1,  # no annotations
            },
        },
    },
    'coco_2017_train': {
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_train2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 118287,
            'bbox': {
                'num_classes': 80,
                'num_instances': 860001,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 860001,
            },
        },
    },
    'coco_2017_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_val2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 80,
                'num_instances': 36781,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 36781,
            },
        },
    },
    'coco_2017_test': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/image_info_test2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 40670,
            'bbox': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
        },
    },
    'coco_2017_test-dev': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/image_info_test-dev2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 20288,
            'bbox': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
        },
    },
    'coco_keypoints_2017_train': {
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/person_keypoints_train2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
        _ANN_FIELDS: {
            'num_images': 118287,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 262465,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 262465,
            },
            'keypoints': {
                'num_classes': 17,
                'num_instances': 262465,
                'flip_map': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
                'connections': (
                    (1, 2), (1, 0), (2, 0), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7),
                    (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (12, 11),
                ),
            },
        },
    },
    'coco_keypoints_2017_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/person_keypoints_val2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 11004,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 11004,
            },
            'keypoints': {
                'num_classes': 17,
                'num_instances': 11004,
                'flip_map': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
                'connections': (
                    (1, 2), (1, 0), (2, 0), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7),
                    (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (12, 11),
                ),
            },
        },
    },
    'coco_keypoints_2017_test': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/image_info_test2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
        _ANN_FIELDS: {
            'num_images': 40670,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'keypoints': {
                'num_classes': 17,
                'num_instances': -1,  # no annotations
                'flip_map': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
                'connections': (
                    (1, 2), (1, 0), (2, 0), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7),
                    (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (12, 11),
                ),
            },
        },
    },
    'coco_keypoints_2017_test-dev': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/image_info_test-dev2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
        _ANN_FIELDS: {
            'num_images': 20288,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'keypoints': {
                'num_classes': 17,
                'num_instances': -1,  # no annotations
                'flip_map': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
                'connections': (
                    (1, 2), (1, 0), (2, 0), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7),
                    (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (12, 11),
                ),
            },
        },
    },
    'coco_panoptic_2017_train': {
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_train2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg', 'panoseg'),
        _ANN_FIELDS: {
            'semseg': {
                'seg_json': _DATA_DIR + '/COCO/annotations/panoptic_train2017.json',
                'seg_root': _DATA_DIR + '/COCO/annotations/panoptic_train2017',
                'ignore_label': 255
            },
        },
    },
    'coco_panoptic_2017_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_val2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg', 'panoseg'),
        _ANN_FIELDS: {
            'semseg': {
                'seg_json': _DATA_DIR + '/COCO/annotations/panoptic_val2017.json',
                'seg_root': _DATA_DIR + '/COCO/annotations/panoptic_val2017',
                'ignore_label': 255
            },
        },
    },
    'coco_seg_2017_train': {
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_train2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 118287,
            'bbox': {
                'num_classes': 80,
                'num_instances': 860001,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 860001,
            },
            'semseg': {
                'num_classes': 92,
                'num_instances': 747458,
                'seg_json': _DATA_DIR + '/COCO/annotations/stuff_train2017.json',
                'seg_root': _DATA_DIR + '/COCO/stuffthingmaps/train2017',
                'semseg_format': 'mask',
            },
        },
    },
    'coco_seg_2017_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_val2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 80,
                'num_instances': 36781,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 36781,
            },
            'semseg': {
                'num_classes': 92,
                'num_instances': 32801,
                'seg_json': _DATA_DIR + '/COCO/annotations/stuff_val2017.json',
                'seg_root': _DATA_DIR + '/COCO/stuffthingmaps/val2017',
                'semseg_format': 'mask',
            },
        },
    },
    'cocohumanparts_personheadface_train': {  # TODO
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/COCOHumanParts/instance_personheadface_train2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
    },
    'cocohumanparts_personheadface_val': {  # TODO
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/COCOHumanParts/instance_personheadface_val2017.json',
        _ANN_TYPES: ('bbox', 'mask', 'keypoints'),
    },
    'objects365v1_train': {
        _IM_DIR: _DATA_DIR + '/Object365v1/images/train',
        _ANN_FN: _DATA_DIR + '/Object365v1/annotations/objects365_train.json',
        _ANN_TYPES: ('bbox',),
        _ANN_FIELDS: {
            'num_images': 608606,
            'bbox': {
                'num_classes': 365,
                'num_instances': 9621875,
            },
        },
    },
    'objects365v1_val': {
        _IM_DIR: _DATA_DIR + '/Object365v1/images/val',
        _ANN_FN: _DATA_DIR + '/Object365v1/annotations/objects365_val.json',
        _ANN_TYPES: ('bbox',),
        _ANN_FIELDS: {
            'num_images': 30000,
            'bbox': {
                'num_classes': 365,
                'num_instances': 479181,
            },
        },
    },
    'CIHP_train': {
        _IM_DIR: _DATA_DIR + '/CIHP/Training/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 28280,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 93213,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 93213,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': 93213,
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/CIHP/Training/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'cihp_united_instance_val': {
        _IM_DIR: _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': 17520,
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/CIHP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'CIHP_test': {
        _IM_DIR: _DATA_DIR + '/CIHP/Testing/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,  # no annotations
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no annotations
                # 'seg_root': _DATA_DIR + '/CIHP/Testing/Category_ids',  # no gt seg
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'LIP_train': {
        _IM_DIR: _DATA_DIR + '/LIP/Training/Images',
        _ANN_FN: _DATA_DIR + '/LIP/annotations/LIP_train.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 30462,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 30462,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 30462,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/LIP/Training/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'LIP_val': {
        _IM_DIR: _DATA_DIR + '/LIP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/LIP/annotations/LIP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 10000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/LIP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'LIP_test': {
        _IM_DIR: _DATA_DIR + '/LIP/Testing/Images',
        _ANN_FN: _DATA_DIR + '/LIP/annotations/LIP_test.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 10000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': -1,  # no annotations
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,  # no annotations
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no annotations
                # 'seg_root': _DATA_DIR + '/LIP/Testing/Category_ids',  # no gt seg
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'ade2017_sceneparsing_train': {
        _IM_DIR: _DATA_DIR + '/ADE2017/images/training',
        _ANN_FN: _DATA_DIR + '/ADE2017/annotations/ade2017_sceneparsing_train.json',
        _ANN_TYPES: ('semseg',),
        _ANN_FIELDS: {
            'num_images': 20210,
            'semseg': {
                'num_classes': 150,  # exclude ignore class
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/ADE2017/annotations_sceneparsing/training',
                'ignore_label': 255,
                'label_shift': -1,
                'semseg_format': 'mask',
            },
        },
    },
    'ade2017_sceneparsing_val': {
        _IM_DIR: _DATA_DIR + '/ADE2017/images/validation',
        _ANN_FN: _DATA_DIR + '/ADE2017/annotations/ade2017_sceneparsing_val.json',
        _ANN_TYPES: ('semseg',),
        _ANN_FIELDS: {
            'num_images': 2000,
            'semseg': {
                'num_classes': 150,  # exclude ignore class
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/ADE2017/annotations_sceneparsing/validation',
                'ignore_label': 255,
                'label_shift': -1,
                'semseg_format': 'mask',
            },
        },
    },
    'ade2017_sceneparsing_test': {
        _IM_DIR: _DATA_DIR + '/ADE2017/images/testing',
        _ANN_FN: _DATA_DIR + '/ADE2017/annotations/ade2017_sceneparsing_test.json',
        _ANN_TYPES: ('semseg',),
        _ANN_FIELDS: {
            'num_images': 3352,
            'semseg': {
                'num_classes': 150,  # exclude ignore class
                'num_instances': -1,  # no annotations
                # 'seg_root': _DATA_DIR + '/ADE2017/annotations_sceneparsing/testing',  # no gt seg
                'ignore_label': 255,
                'label_shift': -1,
                'semseg_format': 'mask',
            },
        },
    },
}


def datasets():
    """Retrieve the list of available dataset names."""
    return COMMON_DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in COMMON_DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return COMMON_DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return COMMON_DATASETS[name][_ANN_FN]


def get_ann_types(name):
    """Retrieve the annotation types for the dataset."""
    return COMMON_DATASETS[name][_ANN_TYPES]


def get_ann_fields(name):
    """Retrieve the annotation fields for the dataset."""
    return COMMON_DATASETS[name].get(_ANN_FIELDS, {})
