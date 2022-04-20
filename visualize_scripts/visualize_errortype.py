# -*- coding: UTF-8 -*-
'''
@Project ：code
@File ：visualize.py
@Author ：jzl
@Date ：2022/3/12 11:41
'''

import argparse
import json
import os
import sys
import cv2
import torch
from tqdm import tqdm
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from torch.cuda.amp import autocast

import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from visualizer import TrackVisualizer

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

YTVIS_CATEGORIES_2021 = {
    1: "airplane",
    2: "bear",
    3: "bird",
    4: "boat",
    5: "car",
    6: "cat",
    7: "cow",
    8: "deer",
    9: "dog",
    10: "duck",
    11: "earless_seal",
    12: "elephant",
    13: "fish",
    14: "flying_disc",
    15: "fox",
    16: "frog",
    17: "giant_panda",
    18: "giraffe",
    19: "horse",
    20: "leopard",
    21: "lizard",
    22: "monkey",
    23: "motorbike",
    24: "mouse",
    25: "parrot",
    26: "person",
    27: "rabbit",
    28: "shark",
    29: "skateboard",
    30: "snake",
    31: "snowboard",
    32: "squirrel",
    33: "surfboard",
    34: "tennis_racket",
    35: "tiger",
    36: "train",
    37: "truck",
    38: "turtle",
    39: "whale",
    40: "zebra",
}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 visualizer")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2021mini/video_maskformer2_R50_bs16_8ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--result-file',
        default='/home/user/Program/video_error/mini360relate/results_minioriginal.json',
        help='path to result json file',
    )

    parser.add_argument(
        "--output",
        default='visualize_output',
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    return parser


def toRLE(mask: object, w: int, h: int):
    """
    Borrowed from Pycocotools:
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """

    if type(mask) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(mask, h, w)
        return mask_util.merge(rles)
    elif type(mask['counts']) == list:
        # uncompressed RLE
        return mask_util.frPyObjects(mask, h, w)
    else:
        return mask


def iou_seq(d_seq, g_seq):
    '''

    :param d_seq: RLE object
    :param g_seq: RLE object
    :return:
    '''
    i = .0
    u = .0
    for d, g in zip(d_seq, g_seq):
        if d and g:
            i += mask_util.area(mask_util.merge([d, g], True))
            u += mask_util.area(mask_util.merge([d, g], False))
        elif not d and g:
            u += mask_util.area(g)
        elif d and not g:
            u += mask_util.area(d)
    # if not u > .0:
    #     print("Mask sizes in video  and category  may not match!")
    iou = i / u if u > .0 else .0
    return iou


def get_predictions_from_json(json_file):
    pre_list = {}
    # fr = {'video_id', 'score', 'category_id', 'segmentations'}
    for fr in tqdm(json_file):

        if fr['video_id'] not in pre_list:
            pre_list[fr['video_id']] = []

        # instance = {"image_size": None, "pred_scores": None, "pred_labels": None,
        #             "pred_masks": None, "instance_id": None}

        vid = fr['video_id']

        fr["instance_id"] = len(pre_list[vid]) + 1

        # instance["pred_scores"] = fr["score"]
        # fr['category_id'] = fr['category_id']
        # mask = [mask_util.decode(_m) for _m in fr['segmentations']]
        # mask = fr['segmentations']
        # instance["pred_masks"] = mask
        # fr['segmentations'] = [mask_util.decode(_m) for _m in fr['segmentations']]
        fr["image_size"] = fr['segmentations'][0]['size']
        pre_list[vid].append(fr)
    return pre_list


def get_groundtruth_from_json(json_file):
    pre_list = {}
    # fr = {'video_id', 'iscrowd', 'height', 'width', 'length', 'segmentations', 'bboxes', 'category_id', 'id', 'areas'}
    for fr in tqdm(json_file):
        if fr['video_id'] not in pre_list:
            pre_list[fr['video_id']] = []
        fr["instance_id"] = len(pre_list[fr['video_id']]) + 1

        # print(fr['segmentations'])
        mask = []
        for seg in fr['segmentations']:
            mask.append(
                toRLE(seg, fr['width'], fr[
                    'height']) if seg != None else
                mask_util.encode(np.array(np.zeros((fr['height'], fr['width']))[:, :, None], order="F", dtype="uint8"))[
                    0])
        fr['rle_segmentations'] = mask

        # fr['segmentations'] = []
        # for _m in fr['rle_segmentations']:
        #     if _m == None:
        #         _m = np.zeros((fr['height'], fr['width']))
        #     fr['segmentations'].append(mask_util.decode(_m))
        ''''''
        fr['segmentations'] = [mask_util.decode(_m) for _m in fr['rle_segmentations']]

        pre_list[fr['video_id']].append(fr)

    return pre_list


def draw_instance_id(im_in, insid, is_gt=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    margin = 5
    thickness = 1
    if is_gt:
        color = (255, 105, 65)
    else:
        color = (13, 23, 227)

    x = y = 0

    for _nt, text in enumerate(insid):
        size = cv2.getTextSize(text, font, font_scale, thickness)

        text_width = size[0][0]
        text_height = size[0][1]

        im_in[y+(margin)*_nt:text_height + margin + y + margin * _nt, :text_width + margin, :] = np.array(
            [220, 220, 220])  # np.zeros((10,10,3))
        x = margin
        y = text_height + y + margin * _nt

        im_in = cv2.putText(np.ascontiguousarray(im_in), text, (x, y), font, font_scale, color, thickness)
    return im_in


if __name__ == '__main__':

    args = get_parser().parse_args()

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    cfg = setup_cfg(args)

    # load jsons and get predictions
    res_json = json.load(open(args.result_file, 'r'))
    print('loading predictions')
    predictions = get_predictions_from_json(res_json)

    # visualize
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    '''
    Metadata(
    evaluator_type='ytvis', 
    image_root='datasets/ytvis_2021_mini/valid/JPEGImages', 
    json_file='datasets/ytvis_2021_mini/valid.json', 
    name='ytvis_2021_mini_val', thing_classes=['airplane', ...], 
    thing_colors=[[106, 0, 228], ...], 
    thing_dataset_id_to_contiguous_id={1: 0,...})
    '''

    video_json = metadata.get('json_file')
    video_json = json.load(open(video_json, 'r'))
    groundtruth = get_groundtruth_from_json(video_json['annotations'])

    error_look_up_dict = {}
    video_example = {'FP': set()}

    for vid, v in enumerate(video_json['videos']):

        # if vid != 39:
        #     continue
        error_look_up_dict['video' + str(vid)] = video_example

        print('processing video', vid)

        '''preprocess'''

        img_path_root = metadata.get('image_root')
        pred = predictions[v['id']]
        anno = groundtruth[v['id']]
        for tmp_g in anno:
            tmp_g['ignore'] = tmp_g['ignore'] if 'ignore' in tmp_g else 0
            tmp_g['ignore'] = 'iscrowd' in tmp_g and tmp_g['iscrowd']
        for tmp_g in anno:
            if tmp_g['ignore']:
                tmp_g['_ignore'] = 1
            else:
                tmp_g['_ignore'] = 0

        '''calculate iou'''

        inds = np.argsort([-d['score'] for d in pred], kind='mergesort')
        dt = [pred[i] for i in inds]
        g = [g['rle_segmentations'] for g in anno]
        gtind = np.argsort([gind['_ignore'] for gind in anno], kind='mergesort')
        g = [g[i] for i in gtind]  # rle gt mask
        gt = [anno[i] for i in gtind]
        d = [d['segmentations'] for d in dt]  # rle dt mask
        iscrowd = [int(o['iscrowd']) for o in gt]

        ious = np.zeros([len(pred), len(anno)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = iou_seq(d[i], g[j])

        '''load frames'''

        vid_frames = []
        for path in v['file_names']:
            path = os.path.join(img_path_root, path)
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        path_root = os.path.join(args.output, 'video_' + str(vid))

        # calucate iou
        # iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        '''match'''

        iouThrs = [0.1, 0.5]

        T = len(iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))

        dtm_01 = np.zeros((T, D))  # iou<0.1 fp
        gtm_index_01 = np.zeros((T, D))
        gtm_01 = np.zeros((T, G))
        dtm_015 = np.zeros((T, D))  # 0.1<iou<0.5 fp
        gtm_index_015 = np.zeros((T, D))
        gtm_015 = np.zeros((T, G))
        dtm_05 = np.zeros((T, D))  # iou>0.5 fp
        gtm_05 = np.zeros((T, G))
        gtm_index_05 = np.zeros((T, D))

        gtm_index = np.zeros((T, D))  # store the index of gt that corespond to dtm

        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        for tind, t in enumerate(iouThrs):
            for dind, d1 in enumerate(dt):

                iou = min([t, 1 - 1e-10])
                m = -1
                m01 = -1
                m015 = -1
                m05 = -1
                for gind, g1 in enumerate(gt):

                    # if ious[dind, gind] < iou:
                    #     if ious[dind, gind] < 0.1:
                    #         if m01 == -1:
                    #             iou01 = ious[dind, gind]
                    #             m01 = gind
                    #         if ious[dind, gind] > iou01:
                    #             iou01 = ious[dind, gind]
                    #             m01 = gind
                    #     else:
                    #         if m015 == -1:
                    #             iou015 = ious[dind, gind]
                    #             m015 = gind
                    #         if ious[dind, gind] > iou015:
                    #             iou015 = ious[dind, gind]
                    #             m015 = gind

                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue

                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break

                    if ious[dind, gind] < iou:
                        continue

                    iou = ious[dind, gind]
                    m = gind

                # if m01 != -1:
                #     dtm_01[tind, dind] = gt[m01]['instance_id']
                #     gtm_index_01[tind, dind] = m01
                #     gtm_01[tind, m01] = d1['instance_id']
                #
                # if m015 != -1:
                #     dtm_015[tind, dind] = gt[m015]['instance_id']
                #     gtm_index_015[tind, dind] = m015
                #     gtm_015[tind, m015] = d1['instance_id']

                if m == -1:
                    # fp
                    for gind, g1 in enumerate(gt):
                        iou = min([t, 1 - 1e-10])

                        tmp_gind = 0
                        tmp_iou = ious[dind, gind]
                        # pick the hightest iou gt for this dt
                        if ious[dind, gind] >= tmp_iou:
                            tmp_iou = ious[dind, gind]
                            tmp_gind = gind
                        if gind != len(gt) - 1:
                            continue

                        # attribute the fp
                        if tmp_iou < iou:  # <0.5   fp

                            if tmp_iou < 0.1:
                                if m01 == -1:
                                    iou01 = tmp_iou
                                    m01 = tmp_gind

                                # if ious[dind, gind] > iou01:
                                #     iou01 = ious[dind, gind]
                                #     m01 = gind

                            else:
                                if m015 == -1:
                                    iou015 = tmp_iou
                                    m015 = tmp_gind

                                # if ious[dind, gind] > iou015:
                                #     iou015 = ious[dind, gind]
                                #     m015 = gind

                        else:  # > 0.5 fp
                            if m05 == -1:
                                iou05 = tmp_iou
                                m05 = tmp_gind
                            # if ious[dind, gind] > iou05:
                            #     iou05 = ious[dind, gind]
                            #     m05 = gind
                    if m01 != -1:
                        dtm_01[tind, dind] = gt[m01]['id']
                        gtm_index_01[tind, dind] = m01
                        gtm_01[tind, m01] = d1['instance_id']

                    if m015 != -1:
                        dtm_015[tind, dind] = gt[m015]['id']
                        gtm_index_015[tind, dind] = m015
                        gtm_015[tind, m015] = d1['instance_id']

                    if m05 != -1:
                        dtm_05[tind, dind] = gt[m05]['id']
                        gtm_index_05[tind, dind] = m05
                        gtm_05[tind, m015] = d1['instance_id']
                    continue

                if d1['category_id'] != gt[m]['category_id']:
                    dtm_05[tind, dind] = gt[m]['id']
                    gtm_index_05[tind, dind] = m
                    gtm_05[tind, m] = d1['instance_id']
                    continue

                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d1['instance_id']
                gtm_index[tind, dind] = m

        dtIg = np.logical_or(dtIg, dtm == 0)

        '''draw gt'''

        print('drawing gt')
        path_gt_root = os.path.join(path_root, 'GT')
        path_gt_root_matched = os.path.join(path_gt_root, 'Matched')
        path_gt_root_missed = os.path.join(path_gt_root, 'Missed')
        image_size = pred[0]["image_size"]
        gt_labels_all = [[], []]  # [[matched],[missed]]
        gt_masks_all = [[], []]
        gt_ins_id_all = [[], []]
        for _a, _gtmind in enumerate(gtm[1]):
            if _gtmind == 0:
                gt_labels_all[1].append(gt[_a]["category_id"] - 1)
                gt_masks_all[1].append(gt[_a]["segmentations"])
                gt_ins_id_all[1].append(gt[_a]["id"])
            else:
                gt_labels_all[0].append(gt[_a]["category_id"] - 1)
                gt_masks_all[0].append(gt[_a]["segmentations"])
                gt_ins_id_all[0].append(gt[_a]["id"])
        for _ri, (gt_labels, gt_masks, gt_ins_id) in enumerate(zip(gt_labels_all, gt_masks_all, gt_ins_id_all)):
            if _ri == 0:
                path_gt_root = path_gt_root_matched
            else:
                path_gt_root = path_gt_root_missed

            for gl, gm, giid in zip(gt_labels, gt_masks, gt_ins_id):
                gt_frame_masks = gm
                # print(gt_frame_masks)
                # cat_id+_+category
                path_gt_root_i = os.path.join(path_gt_root,
                                              'insID-' + str(int(giid)) + '_' + YTVIS_CATEGORIES_2021[gl + 1])
                os.makedirs(path_gt_root_i, exist_ok=True)
                for frame_idx in range(len(vid_frames)):
                    frame = vid_frames[frame_idx][:, :, ::-1]
                    visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                    ins = Instances(image_size)
                    ins.scores = [1]
                    ins.pred_classes = [gl]
                    gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                    ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                    vis_output = visualizer.draw_instance_predictions(predictions=ins)
                    text = ['insID:' + str(int(giid)) + '  ' + YTVIS_CATEGORIES_2021[gl + 1]]
                    vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1], text, is_gt=True)
                    cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.jpg'),
                                vis_im)
                print('successfully saved gt')

        '''draw tp'''

        path_tp_root = os.path.join(path_root, 'TP')
        # tp_match [(dt_index,gt_id)]
        tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm[1]) if _dtm != 0]
        # print(dtm[1])
        # print(tp_match)
        # sys.exit()
        tp_labels = []
        tp_masks = []
        tp_ins_id = []
        tp_matchgt_insid = []  # the gt instance_id that matched to the dt
        tp_matchgt_index = []  # the gt index that matched to the dt
        tp_dt_index = []
        tp_dt_score = []
        for (tpm_i, tpm_dtm) in tp_match:
            tp_labels.append(dt[tpm_i]['category_id'] - 1)
            tp_masks.append(dt[tpm_i]['segmentations'])
            tp_ins_id.append(dt[tpm_i]['instance_id'])
            tp_matchgt_insid.append(tpm_dtm)
            tp_matchgt_index.append(gtm_index[1, tpm_i])
            tp_dt_index.append(tpm_i)
            tp_dt_score.append(dt[tpm_i]['score'])

        image_size = pred[0]["image_size"]
        # tp_labels = [_a["category_id"] - 1 for _a in anno]
        # tp_masks = [_a["segmentations"] for _a in anno]
        # tp_ins_id = [_a["instance_id"] for _a in anno]
        print('--num_tp', len(tp_labels))
        for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id, tp_matchgt_insid,
                                                                   tp_dt_index, tp_matchgt_index, tp_dt_score):
            print('drawing tp')
            gt_frame_masks = [mask_util.decode(_m) for _m in gm]
            # print(gt_frame_masks)
            # iou + _ + cat_id + _ + category + score
            path_gt_root_i = os.path.join(path_tp_root,
                                          'predIdx-' + str(int(giid)) + '_score-' + str(round(tpscore, 2)) +
                                          '_iou-' + str(round(ious[tpidx, int(matchidx)], 2)) + '_' +
                                          YTVIS_CATEGORIES_2021[gl + 1])
            os.makedirs(path_gt_root_i, exist_ok=True)
            for frame_idx in range(len(vid_frames)):
                frame = vid_frames[frame_idx][:, :, ::-1]
                visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                ins = Instances(image_size)
                ins.scores = [tpscore]
                ins.pred_classes = [gl]
                gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                vis_output = visualizer.draw_instance_predictions(predictions=ins)
                text = [
                    'predIdx:' + str(int(giid)) + ' gtID:' + str(int(matchid)) + ' ' + YTVIS_CATEGORIES_2021[gl + 1],
                    'score:' + str(round(tpscore, 2)) + ' iou:' + str(round(ious[tpidx, int(matchidx)], 2)),
                ]
                vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1], text)
                cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.jpg'),
                            vis_im)
            print('successfully saved tp')

        '''draw fp  iou<0.1 | 0.1< iou<0.5 | iou>0.5  '''

        for k in range(3):

            # tp_match [(dt_index,gt_id)]
            # tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm[1]) if _dtm != 0]
            tp_labels = []
            tp_masks = []
            tp_ins_id = []
            tp_matchgt_insid = []  # the gt instance_id that matched to the dt
            tp_matchgt_index = []  # the gt index that matched to the dt
            tp_dt_index = []
            tp_dt_score = []

            # draw normal fp
            # if k == 0:
            #     path_tp_root = os.path.join(path_root, 'fp', 'iou<0.1')
            #     tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_01[1]) if _dtm != 0]
            # elif k == 1:
            #     path_tp_root = os.path.join(path_root, 'fp', '0.1<iou<0.5')
            #     tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_015[1]) if _dtm != 0]
            # else:
            #     path_tp_root = os.path.join(path_root, 'fp', 'iou>0.5')
            #     tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_05[1]) if _dtm != 0]

            # draw bgk error
            if k == 0:
                path_tp_root = os.path.join(path_root, 'FP', 'background')

                tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_01[1]) if _dtm != 0]
                if len(tp_match) > 0:
                    error_look_up_dict['video' + str(vid)]['FP'].add('bkg')

                tp_labels = []
                tp_masks = []
                tp_ins_id = []
                tp_matchgt_insid = []  # the gt instance_id that matched to the dt
                tp_matchgt_index = []  # the gt index that matched to the dt
                tp_dt_index = []
                tp_dt_score = []
                for (tpm_i, tpm_dtm) in tp_match:

                    tp_labels.append(dt[tpm_i]['category_id'] - 1)
                    tp_masks.append(dt[tpm_i]['segmentations'])
                    tp_ins_id.append(dt[tpm_i]['instance_id'])
                    tp_matchgt_insid.append(tpm_dtm)
                    if k == 0:
                        tp_matchgt_index.append(gtm_index_01[1, tpm_i])
                    elif k == 1:
                        tp_matchgt_index.append(gtm_index_015[1, tpm_i])
                    else:
                        tp_matchgt_index.append(gtm_index_05[1, tpm_i])

                    tp_dt_index.append(tpm_i)
                    tp_dt_score.append(dt[tpm_i]['score'])

                image_size = pred[0]["image_size"]
                # tp_labels = [_a["category_id"] - 1 for _a in anno]
                # tp_masks = [_a["segmentations"] for _a in anno]
                # tp_ins_id = [_a["instance_id"] for _a in anno]
                print('--num_fp ' + str(k), len(tp_labels))
                for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id,
                                                                           tp_matchgt_insid,
                                                                           tp_dt_index, tp_matchgt_index,
                                                                           tp_dt_score):
                    print('drawing fp ' + str(k))
                    gt_frame_masks = [mask_util.decode(_m) for _m in gm]
                    # print(gt_frame_masks)
                    # iou + _ + cat_id + _ + category + score
                    path_gt_root_i = os.path.join(path_tp_root,
                                                  'predIdx-' + str(int(giid)) + '_score-' + str(round(tpscore, 2)) +
                                                  '_iou-' + str(round(ious[tpidx, int(matchidx)], 2)) + '_' +
                                                  YTVIS_CATEGORIES_2021[gl + 1])
                    os.makedirs(path_gt_root_i, exist_ok=True)
                    for frame_idx in range(len(vid_frames)):
                        frame = vid_frames[frame_idx][:, :, ::-1]
                        visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                        ins = Instances(image_size)
                        ins.scores = [tpscore]
                        ins.pred_classes = [gl]
                        gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                        ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                        vis_output = visualizer.draw_instance_predictions(predictions=ins)
                        text = [
                            'predIdx:' + str(int(giid)) + ' gtID:' + str(int(matchid)) + ' ' + YTVIS_CATEGORIES_2021[
                                gl + 1],
                            'score:' + str(round(tpscore, 2)) + ' iou:' + str(round(ious[tpidx, int(matchidx)], 2)),
                        ]
                        vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1],
                                                  text)
                        cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.jpg'),
                                    vis_im)
                    print('successfully saved fp ' + str(k))

            # draw spatial and temporal error
            frame_thr = 0.3
            temporal_thr = 0.7
            if k == 1:
                path_tp_root = os.path.join(path_root)
                tp_match_all = [(_i, _dtm, gtm_index_015[1][_i]) for _i, _dtm in enumerate(dtm_015[1]) if _dtm != 0]
                tp_match_cat = [[], []]  # [[spatial],[temporal]]
                for _ii, _dtm, _gt_i in tp_match_all:
                    dt_mask, gt_mask = d[int(_ii)], g[int(_gt_i)]

                    frame_gt_iou = np.zeros(len(dt_mask))  # per frame iou
                    for _i, (_pr, _prgt) in enumerate(zip(dt_mask, gt_mask)):

                        if not np.any(mask_util.decode(_prgt)) and not np.any(mask_util.decode(_pr)):
                            # gt and pred both have no mask
                            tmp_fiou = 1.0
                        # elif not np.any(mask_util.decode(_prgt)) and np.any(mask_util.decode(_pr)):
                        #     tmp_fiou = 0.0
                        else:
                            tmp_fiou = mask_util.iou([_pr], [_prgt], [False])

                        frame_gt_iou[_i] = tmp_fiou
                    temporal_good = 0
                    for _iou in frame_gt_iou:
                        if _iou > frame_thr:
                            temporal_good += 1

                    temporal_overlap = temporal_good / len(frame_gt_iou)
                    # print(temporal_overlap)

                    # Test for SpacialBadError
                    if temporal_overlap >= temporal_thr:
                        tp_match_cat[0].append((_ii, _dtm))

                    # Test for TemporalBadError
                    elif temporal_overlap < temporal_thr:
                        tp_match_cat[1].append((_ii, _dtm))

                path_tp_root_s = os.path.join(path_tp_root, 'FP', 'SpacialBadError')
                path_tp_root_t = os.path.join(path_tp_root, 'FP', 'TemporalBadError')
                for tpc_i, tp_match in enumerate(tp_match_cat):
                    # print('------------------------------', len(tp_match))
                    if len(tp_match) == 0:
                        continue
                    if tpc_i == 0:
                        path_tp_root = path_tp_root_s
                        error_look_up_dict['video' + str(vid)]['FP'].add('SpacialBadError')
                    else:
                        path_tp_root = path_tp_root_t
                        error_look_up_dict['video' + str(vid)]['FP'].add('TemporalBadError')

                    tp_labels = []
                    tp_masks = []
                    tp_ins_id = []
                    tp_matchgt_insid = []  # the gt instance_id that matched to the dt
                    tp_matchgt_index = []  # the gt index that matched to the dt
                    tp_dt_index = []
                    tp_dt_score = []
                    for (tpm_i, tpm_dtm) in tp_match:

                        tp_labels.append(dt[tpm_i]['category_id'] - 1)
                        tp_masks.append(dt[tpm_i]['segmentations'])
                        tp_ins_id.append(dt[tpm_i]['instance_id'])
                        tp_matchgt_insid.append(tpm_dtm)
                        if k == 0:
                            tp_matchgt_index.append(gtm_index_01[1, tpm_i])
                        elif k == 1:
                            tp_matchgt_index.append(gtm_index_015[1, tpm_i])
                        else:
                            tp_matchgt_index.append(gtm_index_05[1, tpm_i])

                        tp_dt_index.append(tpm_i)
                        tp_dt_score.append(dt[tpm_i]['score'])

                    image_size = pred[0]["image_size"]
                    # tp_labels = [_a["category_id"] - 1 for _a in anno]
                    # tp_masks = [_a["segmentations"] for _a in anno]
                    # tp_ins_id = [_a["instance_id"] for _a in anno]
                    print('--num_fp ' + str(k), len(tp_labels))
                    for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id,
                                                                               tp_matchgt_insid,
                                                                               tp_dt_index, tp_matchgt_index,
                                                                               tp_dt_score):
                        print('drawing fp ' + str(k))
                        gt_frame_masks = [mask_util.decode(_m) for _m in gm]
                        # print(gt_frame_masks)
                        # iou + _ + cat_id + _ + category + score
                        path_gt_root_i = os.path.join(path_tp_root,
                                                      'predIdx-' + str(int(giid)) + '_score-' + str(round(tpscore, 2)) +
                                                      '_iou-' + str(round(ious[tpidx, int(matchidx)], 2)) + '_' +
                                                      YTVIS_CATEGORIES_2021[gl + 1])
                        os.makedirs(path_gt_root_i, exist_ok=True)
                        for frame_idx in range(len(vid_frames)):
                            frame = vid_frames[frame_idx][:, :, ::-1]
                            visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                            ins = Instances(image_size)
                            ins.scores = [tpscore]
                            ins.pred_classes = [gl]
                            gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                            ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                            vis_output = visualizer.draw_instance_predictions(predictions=ins)
                            text = [
                                'predIdx:' + str(int(giid)) + ' gtID:' + str(int(matchid)) + ' ' +
                                YTVIS_CATEGORIES_2021[gl + 1],
                                'score:' + str(round(tpscore, 2)) + ' iou:' + str(round(ious[tpidx, int(matchidx)], 2)),
                            ]
                            vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1],
                                                      text)
                            cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.jpg'),
                                        vis_im)
                        print('successfully saved fp ' + str(k))

            # draw cls and dup error
            if k == 2:
                path_tp_root_2 = os.path.join(path_root, 'FP')
                tp_match_all = [(_i, _dtm) for _i, _dtm in enumerate(dtm_05[1]) if _dtm != 0]
                tp_match_cat = [[], []]  # [[cls],[dup]]
                for (_i, _dtm) in tp_match_all:
                    if dt[_i]['category_id'] != gt[int(gtm_index_05[1, _i])]['category_id']:
                        tp_match_cat[0].append((_i, _dtm))
                    else:
                        tp_match_cat[1].append((_i, _dtm))
                path_tp_root_cls = os.path.join(path_tp_root_2, 'cls')
                path_tp_root_dup = os.path.join(path_tp_root_2, 'dup')

                for _ii, tp_match in enumerate(tp_match_cat):
                    if len(tp_match) == 0:
                        continue
                    if _ii == 0:
                        path_tp_root = path_tp_root_cls
                        error_look_up_dict['video' + str(vid)]['FP'].add('cls')
                    else:
                        path_tp_root = path_tp_root_dup
                        error_look_up_dict['video' + str(vid)]['FP'].add('dup')
                    tp_labels = []
                    tp_masks = []
                    tp_ins_id = []
                    tp_matchgt_insid = []  # the gt instance_id that matched to the dt
                    tp_matchgt_index = []  # the gt index that matched to the dt
                    tp_dt_index = []
                    tp_dt_score = []
                    for (tpm_i, tpm_dtm) in tp_match:

                        tp_labels.append(dt[tpm_i]['category_id'] - 1)
                        tp_masks.append(dt[tpm_i]['segmentations'])
                        tp_ins_id.append(dt[tpm_i]['instance_id'])
                        tp_matchgt_insid.append(tpm_dtm)
                        if k == 0:
                            tp_matchgt_index.append(gtm_index_01[1, tpm_i])
                        elif k == 1:
                            tp_matchgt_index.append(gtm_index_015[1, tpm_i])
                        else:
                            tp_matchgt_index.append(gtm_index_05[1, tpm_i])

                        tp_dt_index.append(tpm_i)
                        tp_dt_score.append(dt[tpm_i]['score'])

                    image_size = pred[0]["image_size"]
                    # tp_labels = [_a["category_id"] - 1 for _a in anno]
                    # tp_masks = [_a["segmentations"] for _a in anno]
                    # tp_ins_id = [_a["instance_id"] for _a in anno]
                    print('--num_fp ' + str(k), len(tp_labels))
                    for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id,
                                                                               tp_matchgt_insid,
                                                                               tp_dt_index, tp_matchgt_index,
                                                                               tp_dt_score):
                        print('drawing fp ' + str(k))
                        gt_frame_masks = [mask_util.decode(_m) for _m in gm]
                        # print(gt_frame_masks)
                        # iou + _ + cat_id + _ + category + score
                        path_gt_root_i = os.path.join(path_tp_root,
                                                      'predIdx-' + str(int(giid)) + '_score-' + str(round(tpscore, 2)) +
                                                      '_iou-' + str(round(ious[tpidx, int(matchidx)], 2)) + '_' +
                                                      YTVIS_CATEGORIES_2021[gl + 1])
                        os.makedirs(path_gt_root_i, exist_ok=True)
                        for frame_idx in range(len(vid_frames)):
                            frame = vid_frames[frame_idx][:, :, ::-1]
                            visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                            ins = Instances(image_size)
                            ins.scores = [tpscore]
                            ins.pred_classes = [gl]
                            gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                            ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                            vis_output = visualizer.draw_instance_predictions(predictions=ins)
                            text = [
                                'predIdx:' + str(int(giid)) + ' gtID:' + str(int(matchid)) + ' ' +
                                YTVIS_CATEGORIES_2021[gl + 1],
                                'score:' + str(round(tpscore, 2)) + ' iou:' + str(round(ious[tpidx, int(matchidx)], 2)),
                            ]
                            vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1],
                                                      text)
                            cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.jpg'),
                                        vis_im)
                        print('successfully saved fp ' + str(k))


    for ks in error_look_up_dict.keys():
        error_look_up_dict[ks]['FP'] = list(error_look_up_dict[ks]['FP'])
    js_save_path = args.output
    json.dump(error_look_up_dict, open(os.path.join(js_save_path, 'error_look_up_dict.json'), 'w'))
