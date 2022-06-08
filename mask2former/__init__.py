# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.mask_former_parsing_instance_dataset_mapper import (
    MaskFormerParsingInstanceDatasetMapper,
)

from .data.dataset_mappers.mask_former_parsing_semantic_dataset_mapper import (
    MaskFormerParsingSemanticDatasetMapper,
)

from .data.dataset_mappers.mask_former_single_parsing_test_dataset_mapper import (
    MaskFormerSingleParsingTestDatasetMapper,
)

from .data.dataset_mappers.mask_former_parsing_instance_lsj_dataset_mapper import (
    MaskFormerParsingInstanceLSJDatasetMapper,
)

from .data.build import build_detection_test_loader


# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA, ParsingWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.parsing_evaluation import ParsingEvaluator
