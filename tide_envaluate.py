# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：tide_envaluate.py
@Author ：jzl
@Date ：2022/3/13 14:39 
'''

from tidecv import TIDE
import tidecv.datasets as datasets

gt = datasets.YTVIS2021(path='../ytvis2021mini/valid_mini.json')
mask_results = datasets.YTVIS2021Result(path='./output/inference/results_minioriginal.json')
tide = TIDE()

tide.evaluate_range(gt,mask_results,mode=TIDE.MASK)
tide.summarize()
tide.plot(out_dir='./tide_output')
