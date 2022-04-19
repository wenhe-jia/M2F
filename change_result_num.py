# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：change_result_num.py
@Author ：jzl
@Date ：2022/4/19 13:41 
'''
import os, sys, json
import numpy
import numpy as np

if __name__ == '__main__':
    result_path = 'output/results.json'
    # dict_keys(['video_id', 'score', 'category_id', 'segmentations'])
    source_json = json.load(open(result_path, 'r'))

    top = 10
    output = []
    # attribute by video
    video_dict = {}
    vid_list = []
    for res in source_json:
        vid = res['video_id']
        if vid not in video_dict:
            video_dict[vid] = [[], []]  # [[res],[scores]]
            vid_list.append(vid)
        video_dict[vid][0].append(res)
        video_dict[vid][1].append(res['score'])
    print(vid_list)
    for v in vid_list:
        reses, scores = video_dict[v]
        score_index = np.argsort(-np.array(scores))
        for i in score_index[:top]:
            output.append(reses[i])
        print(len(reses))

    json.dump(output, open(os.path.join('output', 'results' + str(top) + '.json'),'w'))
