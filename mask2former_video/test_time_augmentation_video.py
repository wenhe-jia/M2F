# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import gc
import logging
import os.path
import sys
import time
from itertools import count

import cv2
import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from pycocotools import mask as maskUtils

from detectron2.data.detection_utils import read_image
from .modeling.test_time_augmentation import DatasetMapperTTA_video
from .utils.memory import retry_if_cuda_oom

__all__ = [
    "SemanticSegmentorWithTTA_video",
]


class SemanticSegmentorWithTTA_video(nn.Module):
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
            tta_mapper = DatasetMapperTTA_video(cfg)

        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        # x={'height': 720, 'width': 1280, 'length': 36, 'video_id': 1, 'image': [tensor,...],'instances': [], 'file_names': []}
        result = retry_if_cuda_oom(self._inference_one_video)(batched_inputs[0])

        return result

    def _flip_final_predictions(self, pred):
        '''

        :param pred(list): perd_masks(tensor)
        :return(list): fliped_masks(tensor)
        '''
        out_pred = []
        for p in pred:
            out_pred.append(p.flip(dims=[2]))

        return out_pred

    def _inference_one_video(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """

        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        final_scores = None
        final_labels = None
        count_predictions = 0
        for inputt, tfm in zip(augmented_inputs, tfms):  # one input for one video
            count_predictions += 1
            with torch.no_grad():
                out = self.model([inputt], use_TTA=True)
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions = self._flip_final_predictions(out.pop("pred_masks"))
                    else:
                        final_predictions = out.pop("pred_masks")
                    final_labels = out.pop("pred_labels")
                    final_scores = out.pop("pred_scores")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        pred_tmp = self._flip_final_predictions(out.pop("pred_masks"))
                    else:
                        pred_tmp = out.pop("pred_masks")
                    final_predictions += pred_tmp
                    final_labels += out.pop("pred_labels")
                    final_scores += out.pop("pred_scores")
        del augmented_inputs
        gc.collect()

        # attribute the result by category
        cat_dict = {}
        for l_ind, l in enumerate(final_labels):
            if l not in cat_dict.keys():
                cat_dict[l] = {'masks': [], 'scores': []}
            cat_dict[l]['masks'].append(final_predictions[l_ind])
            cat_dict[l]['scores'].append(final_scores[l_ind])

        del final_predictions
        gc.collect()

        # use nms to merge the resuts
        cat_dict = self.mask_NMS(cat_dict, IOU_thr=0.65)

        # transform to output list
        final_scores = []
        final_predictions = []
        final_labels = []
        for k, v in cat_dict.items():
            for _ in range(len(v['scores'])):
                final_labels.append(k)
            final_scores += list(v.pop('scores'))
            final_predictions += list(v.pop('masks'))

        del cat_dict
        gc.collect()

        final_predictions = [pre > 0 for pre in final_predictions]

        return {"image_size": orig_shape,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
                "pred_masks": final_predictions,
                }

    def _get_augmented_inputs(self, input):
        # video_mapper
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _iou_seq(self, d_seq, g_seq):
        '''

        :param d_seq: RLE object
        :param g_seq: RLE object
        :return:
        '''
        i = .0
        u = .0
        for d, g in zip(d_seq, g_seq):
            if d and g:
                i += maskUtils.area(maskUtils.merge([d, g], True))
                u += maskUtils.area(maskUtils.merge([d, g], False))
            elif not d and g:
                u += maskUtils.area(g)
            elif d and not g:
                u += maskUtils.area(d)
        # if not u > .0:
        #     print("Mask sizes in video  and category  may not match!")
        iou = i / u if u > .0 else .0
        return iou

    def mask_merge(self, masks_in):
        '''

        :param masks_in:(list)
        :return: merged mask
        '''
        ms_tmp = None
        for i, ms in enumerate(masks_in):
            if i == 0:
                ms_tmp = ms
            else:
                ms_tmp += ms

        return ms_tmp / len(masks_in)

    def score_merge(self, scores_in):
        '''

        :param scores_in: (list)
        :return: merged score
        '''
        return np.max(scores_in)

    def msk2rle(self, maskin):
        '''

        :param mask:(frames,h,w)
        :return:
        '''
        maskin = maskin > 0
        _rle = [
            maskUtils.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in maskin
        ]
        for rle in _rle:
            rle["counts"] = rle["counts"].decode("utf-8")
        return _rle

    def mask_NMS(self, catdict_in, IOU_thr=0.65):
        '''

        :param catdict_in: input attributed by category
        :return: dict processed by nms
        '''

        for k, v in catdict_in.items():
            # sort the values by scores
            if len(v['scores']) > 1:
                index_s = np.argsort(-np.asarray(v['scores']))
                v['scores'] = np.asarray(v['scores'])[index_s]
                v['masks'] = [v['masks'][s] for s in index_s]
                v['rle'] = [self.msk2rle(r) for r in v['masks']]

                # apply nms
                supressed = np.zeros(len(v['masks']))
                score_m = []
                mask_m = []
                for i in range(len(v['masks'])):
                    if supressed[i] == 1:
                        continue
                    keep = [i]

                    # mask to rle
                    rle_1 = v['rle'][i]

                    if i != len(v['masks']) - 1:
                        for j in range(i + 1, len(v['masks'])):
                            if supressed[j] == 1:
                                continue

                            # mask to rle
                            rle_2 = v['rle'][j]

                            iou = self._iou_seq(rle_1, rle_2)
                            # print('iou', iou)
                            if iou >= IOU_thr:
                                supressed[j] = 1
                                keep.append(j)
                    score_m.append(self.score_merge(v['scores'][np.asarray(keep)]))

                    mask_m.append(self.mask_merge([v['masks'][s] for s in keep]))

                v['scores'] = score_m
                v['masks'] = mask_m
        return catdict_in
