# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder

from ..solov2.heads import SOLOv2InsHead, SOLOv2MaskHead


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        # image insseg
        solo_ins_head_cfg = {
            "num_classes": num_classes,
            "num_kernels": 256,  # cfg.MODEL.SOLOV2.NUM_KERNELS
            "num_grids": [36, 24, 16],  # cfg.MODEL.SOLOV2.NUM_GRIDS, [40, 36, 24, 16, 12]
            "instance_in_features": ["p3", "p4", "p5"],
            # cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES, ["p2", "p3", "p4", "p5", "p6"]
            "instance_strides": [8, 16, 32],  # cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES, [8, 8, 16, 32, 32]
            "instance_in_channels": 256,  # cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
            "instance_channels": 512,  # # cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
            "type_dcn": "DCN",  # Convolutions to use in the towers, cfg.MODEL.SOLOV2.TYPE_DCN
            "num_levels": 3,  # len(instance_in_features)
            "num_instance_convs": 4,  # cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS
            "use_dcn_in_instance": False,  # cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE
            "use_coord_conv": True,  # cfg.MODEL.SOLOV2.USE_COORD_CONV
            "norm": "GN",  # None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM  # 'GN'
            "prior_prob": 0.01  # cfg.MODEL.SOLOV2.PRIOR_PROB
        }
        solo_mask_head_cfg = {
            "mask_on": True,  # cfg.MODEL.MASK_ON
            "num_masks": 256,  # # cfg.MODEL.SOLOV2.NUM_MASKS
            "mask_in_features": ["p3", "p4", "p5"],  # cfg.MODEL.SOLOV2.MASK_IN_FEATURES, ["p2", "p3", "p4", "p5"]
            "mask_in_channels": 256,  # cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
            "mask_channels": 128,  # cfg.MODEL.SOLOV2.MASK_CHANNELS
            "num_levels": 3,  # len(mask_in_features)
            "norm": "GN"  # None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM,  'GN'
        }

        self.solo_ins_head = SOLOv2InsHead(solo_ins_head_cfg)
        self.solo_mask_head = SOLOv2MaskHead(solo_mask_head_cfg)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, image_sizes, targets, mask=None):
        return self.layers(features, image_sizes, targets, mask)

    def layers(self, features, image_sizes, targets, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        # solov2 image insseg
        ms_features_reverse = multi_scale_features.reverse()  # [res5(1/32), res4(1/16), res3(1/8)] --> [res3(1/8), res4(1/16), res5(1/32)]
        solov2_cate_pred, solov2_kernel_pred = self.ins_head(ms_features_reverse)
        solov2_mask_pred = self.mask_head(ms_features_reverse)
        selected_query = self.get_query(solov2_cate_pred, solov2_kernel_pred, solov2_mask_pred, image_sizes, targets)



        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions

    def get_query(self, pred_cates, pred_kernels, pred_masks, cur_sizes, images):
        return None
