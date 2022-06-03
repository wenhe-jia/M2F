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
    'cihp_semseg_val': {
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
                'flip_map': ((13, 14), (15, 16), (17, 18)),
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
    'cihp_part_instance_val': {
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
    'lip_semseg_val': {
        _IM_DIR: _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
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
                'num_instances': -1,
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
    'lip_part_instance_val': {
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
    'lip_united_instance_val': {
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
    "lip_instance_val": {
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
