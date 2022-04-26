# -*- coding: UTF-8 -*-
'''
@Project ：Mask2Former 
@File ：test_parsing_eval.py
@Author ：jzl
@Date ：2022/4/24 9:16 
'''
import os
import sys
from collections import Counter

import cv2
import numpy as np
from mask2former.evaluation.parsing_evaluation import ParsingEvaluator
from mask2former.evaluation.parsing_eval_api import ParsingGT
from mask2former.evaluation.parsing_eval_api.utils import get_parsing

if __name__ == '__main__':
    evaluater = ParsingEvaluator('cihp_semseg_val', ('parsing',), False, output_dir='./eval_output')
    evaluater.reset()
    pars_gt = evaluater.parsing_GT
    pseudo_dt = []
    root='/home/user/Program/m2f/Mask2Former/stored_output/inference/global_parsing'
    for im_id in pars_gt.ids:
        im_id-=1
        input = [{'image_id': im_id}]
        parsing_ids = np.unique(cv2.imread(os.path.join(root,pars_gt.get_img_info(im_id)['file_name'])))
        parsing = get_parsing(root, pars_gt.get_img_info(im_id)['file_name'], parsing_ids)
        outputs = []
        p_num = [0] * 19
        for p in parsing:
            for _p in p:
                for _pp in dict(Counter(_p.tolist())).keys():
                    if _pp != 0:
                        p_num[_pp-1] = 1
            outputs.append({'category_id': 1, 'parsing': p, 'score': 1, 'instance_scores': 1, 'part_scores': p_num
                               , 'parsing_bbox_scores': 1})
        evaluater.process(input, outputs)

    evaluater.evaluate()



    # evaluater = ParsingEvaluator('cihp_semseg_val', ('parsing',), False, output_dir='./eval_output')
    # evaluater.reset()
    # pars_gt = evaluater.parsing_GT
    # pseudo_dt = []
    # for im_id in pars_gt.ids:
    #     im_id-=1
    #     input = [{'image_id': im_id}]
    #     parsing_ids = [obj['parsing_id'] for obj in pars_gt.pull_target(im_id)]
    #     parsing = get_parsing(pars_gt.root, pars_gt.get_img_info(im_id)['file_name'], parsing_ids)
    #     outputs = []
    #     p_num = [0] * 19
    #     for p in parsing:
    #         for _p in p:
    #             for _pp in dict(Counter(_p.tolist())).keys():
    #                 if _pp != 0:
    #                     p_num[_pp-1] = 1
    #         outputs.append({'category_id': 1, 'parsing': p, 'score': 1, 'instance_scores': 1, 'part_scores': p_num
    #                            , 'parsing_bbox_scores': 1})
    #     evaluater.process(input, outputs)
    #
    # evaluater.evaluate()
