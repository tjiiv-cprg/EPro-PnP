"""
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: ref.py
@time: 18-10-24 下午9:00
@desc:
"""

import numpy as np
import os

# path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..') 
dataset_dir = os.path.join(root_dir, 'dataset')
exp_dir = os.path.join(root_dir, 'exp')
cache_dir = os.path.join(root_dir, 'dataset_cache')

# background images
bg_dir = os.path.join(dataset_dir, 'bg_images')

# linemod dataset
lm_dir = os.path.join(dataset_dir, 'lm')
lm_model_dir = os.path.join(lm_dir, 'models')
lm_model_info_pth = os.path.join(lm_dir, 'models', 'models_info.txt')
lm_train_imgn_dir = os.path.join(lm_dir, 'imgn')
lm_train_real_dir = os.path.join(lm_dir, 'real_train')
lm_test_dir = os.path.join(lm_dir, 'real_test')

# objects
lm_obj = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
lmo_obj = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
idx2obj = {
             1: 'ape',
             2: 'benchvise',
             3: 'bowl',
             4: 'camera',
             5: 'can',
             6: 'cat',
             7: 'cup',
             8: 'driller',
             9: 'duck',
             10: 'eggbox',
             11: 'glue',
             12: 'holepuncher',
             13: 'iron',
             14: 'lamp',
             15: 'phone'
             }
obj_num = len(idx2obj)
def obj2idx(obj_name):
    for k, v in idx2obj.items():
        if v == obj_name:
            return k

# camera
im_w = 640
im_h = 480
im_c = (im_h / 2, im_w / 2)
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
