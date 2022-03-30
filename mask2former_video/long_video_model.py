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
from detectron2.utils.logger import _log_api_usage

__all__ = [
    "LongVideo_inference_model",
]




class LongVideo_inference_model(nn.Module):
    def __init__(self, cfg, model):
        '''

        Args:
            cfg(CfgNode):
            model: model to inference long video
        '''
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.model = model
        _log_api_usage("longvideo." + self.__class__.__name__)

    def __call__(self, inputs, num_clips=2, use_TTA=False):
        '''

        Args:
            inputs(dict):[dict]
            num_clips(int):

        Returns:

        '''
        # print(inputs[0]['image'])
        logger = logging.getLogger(__name__)
        self.use_TTA = use_TTA
        self.model.train(self.training)
        # x = {'height': 720, 'width': 1280, 'length': 36, 'video_id': 1, 'image': [tensor, ...], 'instances': [],
        #      'file_names': []}

        for n_c in range(1, 11):
            try:
                clips = self._divide_videos(inputs[0], n_c)
                result = self._inference_clips(clips)
                logger.info('inference on {} clips'.format(n_c))
                break
            except RuntimeError as e:
                if "CUDA out of memory. " in str(e):
                    assert n_c < 10, 'cuda oom occurs even inference on 10 clips'
                    continue

        # clips = self._divide_videos(inputs[0], num_clips)
        # result = self._inference_clips(clips)
        return result

    def _inference_clips(self, clips_in: list, on_gpu=True):
        '''

        Args:
            clips_in(list):

        Returns:

        '''
        output_dict = {}
        for clip_index, clip in enumerate(clips_in):
            # model_out={
            #     "image_size": (output_height, output_width),
            #     "pred_scores": [int*10],
            #     "pred_labels": [int*10],
            #     "pred_masks": [tensor*10],
            #     'qurry_feature': (10,C)
            # }
            model_out = self.model([clip], self.use_TTA, on_gpu)
            self.device = model_out['qurry_feature'].device

            if clip_index == 0:
                output_dict = model_out
            else:
                output_dict = self._merge_outputs(output_dict, model_out)

        '''copy to cpu'''
        output_dict['pred_masks'] = [_m.cpu() for _m in output_dict['pred_masks']]
        output_dict.pop("qurry_feature")
        return output_dict

    def _merge_outputs(self, out_before, out_now, dis_thr=5):
        '''

        Args:
            out_before(dict):
            out_now(dict):

        Returns:

        '''

        # def save_images(final_labels, final_scores, final_predictions, fag):
        #     final_predictions = [f > 0 for f in final_predictions]
        #
        #     path_dir = './output_image'
        #     for i, (l, s, m) in enumerate(zip(final_labels, final_scores, final_predictions)):
        #         dir_name = os.path.join(path_dir, 'video' + str(fag),
        #                                 str(l) + '_' + str(s))
        #         if not os.path.exists(dir_name):
        #             os.makedirs(dir_name)
        #         for j, ms in enumerate(m):
        #             cv2.imwrite(os.path.join(dir_name, str(j) + '.png'), np.asarray(ms).astype(np.float) * 255)
        #             print('save ' + os.path.join(dir_name, str(j) + '.png'))
        #
        # save_images(out_now['pred_labels'], out_now['pred_scores'], out_now['pred_masks'], vid)
        # return {
        #     "image_size": out_before['image_size'],
        #     "pred_scores": [],
        #     "pred_labels": [],
        #     "pred_masks": [],
        # }

        out1 = self.attribute_by_category(out_before)
        out2 = self.attribute_by_category(out_now)

        final_out = {
            "image_size": out_before['image_size'],
            "pred_scores": [],
            "pred_labels": [],
            "pred_masks": [],
            "qurry_feature": [],
        }
        em_frame = torch.full((out_before['pred_masks'][0][0].shape[0], out_before['pred_masks'][0][0].shape[1]),
                              False).to(self.device)
        empty_mask2 = em_frame.repeat(len(out_before['pred_masks'][0]), 1, 1)
        empty_mask1 = em_frame.repeat(len(out_now['pred_masks'][0]), 1, 1)

        no_exist_cat = []
        for k1, v1 in out1.items():
            if k1 not in out2:
                no_exist_cat.append(k1)
                continue
            v2 = out2.pop(k1)

            # 1.bbox center match

            box_center_dist = torch.zeros((len(v1['masks']), len(v2['masks']))).to(self.device)
            for mind1, m1 in enumerate(v1['masks']):
                boxes1 = masks_to_boxes(m1)
                centr1 = torch.as_tensor(
                    [torch.mean(boxes1[-1][0::2]), torch.mean(boxes1[-1][1::2])])  # last clip's last frame
                for mind2, m2 in enumerate(v2['masks']):
                    boxes2 = masks_to_boxes(m2)
                    centr2 = torch.as_tensor(
                        [torch.mean(boxes2[0][0::2]), torch.mean(boxes2[0][1::2])])  # current clip's first frame
                    box_center_dist[mind1, mind2] = 0.05 * torch.linalg.norm(
                        centr1 - centr2) + 0.95 * torch.linalg.norm(
                        v1['qurry_feature'][mind1] - v2['qurry_feature'][mind2])

            box_center_dist[torch.where(box_center_dist > 65)] = 0

            match1 = torch.zeros(box_center_dist.shape[0]).to(self.device)
            match2 = torch.zeros(box_center_dist.shape[1]).to(self.device)
            m_dis = torch.where(box_center_dist > 0)  # [[x,y],[]...]
            m_dis = torch.stack([m_dis[0], m_dis[1]], dim=0).T
            # print(m_dis.T)
            dis_value = [box_center_dist[_v[0], _v[1]] for _v in m_dis]
            top_index = torch.argsort(torch.as_tensor(dis_value))

            for m_d in m_dis[top_index]:
                if match1[m_d[0]] == 1 or match2[m_d[1]] == 1:  # already being matched
                    continue
                final_out["pred_scores"].append(max(v1['scores'][m_d[0]], v2['scores'][m_d[1]]))
                final_out["pred_masks"].append(torch.cat([v1['masks'][m_d[0]], v2['masks'][m_d[1]]], 0))
                final_out["pred_labels"].append(k1)
                final_out["qurry_feature"].append(v2['qurry_feature'][m_d[1]])
                match1[m_d[0]] = match2[m_d[1]] = 1

            for i, mc1 in enumerate(match1):
                if mc1 == 1:
                    continue
                final_out["pred_scores"].append(v1['scores'][i])
                # print(v1['masks'][i].shape)
                final_out["pred_masks"].append(torch.cat([v1['masks'][i], empty_mask1], 0))
                final_out["pred_labels"].append(k1)
                final_out["qurry_feature"].append(v1['qurry_feature'][i])
            for i, mc2 in enumerate(match2):
                if mc2 == 1:
                    continue
                final_out["pred_scores"].append(v2['scores'][i])
                final_out["pred_masks"].append(torch.cat([empty_mask2, v2['masks'][i]], 0))
                final_out["pred_labels"].append(k1)
                final_out["qurry_feature"].append(v2['qurry_feature'][i])
        for no_cat in no_exist_cat:
            for i, mo1 in enumerate(out1[no_cat]['masks']):
                final_out["pred_scores"].append(out1[no_cat]['scores'][i])
                final_out["pred_masks"].append(torch.cat([mo1, empty_mask1], 0))
                final_out["pred_labels"].append(no_cat)
                final_out["qurry_feature"].append(out1[no_cat]['qurry_feature'][i])

        for k2, v2 in out2.items():
            for i, mo2 in enumerate(out2[k2]['masks']):
                final_out["pred_scores"].append(out2[k2]['scores'][i])
                final_out["pred_masks"].append(torch.cat([empty_mask2, mo2], 0))
                final_out["pred_labels"].append(k2)
                final_out["qurry_feature"].append(out2[k2]['qurry_feature'][i])

        indx = torch.argsort(-torch.as_tensor(final_out["pred_scores"]))[:10]

        final_out["pred_scores"] = [final_out["pred_scores"][_m] for _m in indx]
        # list(torch.as_tensor(final_out["pred_scores"])[indx])
        final_out["pred_masks"] = [final_out["pred_masks"][_m] for _m in indx]
        final_out["pred_labels"] = [final_out["pred_labels"][_m] for _m in indx]
        # list(torch.as_tensor(final_out["pred_labels"])[indx])
        # print((final_out["pred_labels"]), (final_out["pred_scores"]))
        # sys.exit()
        return final_out

    # qf1 = np.asarray(out_before['qurry_feature'])
    # qf2 = np.asarray(out_now['qurry_feature'])
    # # match1 = np.zeros(10)
    # # match2 = np.zeros(10)
    # match = np.zeros((10, 10))
    # for qind1, q1 in enumerate(qf1):
    #     for qind2, q2 in enumerate(qf2):
    #         match[qind1, qind2] = np.linalg.norm(q1 - q2)
    # match = match * (match < dis_thr)
    # match[np.where(match == 0)] = np.inf
    #
    # print(match)
    #
    # for m in match:
    #     i = np.where(m == np.min(m))
    #     print(i)
    # sys.exit()

    def _divide_videos(self, video_in, num):
        '''

        Args:
            video_in(dict):input video dict
            num(int):how many clips you want to divide

        Returns: divided video clip dict list

        '''
        assert num < video_in['length']
        len_clip = video_in['length'] // num

        clip_list = []
        # print(video_in.keys())
        video = video_in['image']
        for i in range(num):
            clip_i = copy.deepcopy(video_in)
            if i != num - 1:
                clip_i['image'] = video[i * len_clip:(i + 1) * len_clip]
                clip_i['length'] = len_clip
            else:
                clip_i['image'] = video[i * len_clip:]
                clip_i['length'] = len(video[i:])
            clip_list.append(clip_i)
            # print('----clip', clip_i['height'], clip_i['width'])
        # print('video_len', video_in['length'])
        # print('--clip_len', [len(_c['image']) for _c in clip_list])
        return clip_list

    def attribute_by_category(self, outdict_in):
        cat_dict = {}
        for l_ind, l in enumerate(outdict_in['pred_labels']):
            if l not in cat_dict.keys():
                cat_dict[l] = {'masks': [], 'scores': [], 'qurry_feature': []}
            cat_dict[l]['masks'].append(outdict_in['pred_masks'][l_ind])
            cat_dict[l]['scores'].append(outdict_in['pred_scores'][l_ind])
            cat_dict[l]['qurry_feature'].append(outdict_in['qurry_feature'][l_ind])

        return cat_dict


def masks_to_boxes(masks_in):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    masks = torch.as_tensor(masks_in) > 0
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
