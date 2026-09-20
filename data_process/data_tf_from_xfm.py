#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/18 17:15
# @Author  : zhongyu
# @Site    : 
# @File    : data_tf_from_xfm.py

'''
from dataset import make_ds
from util.dpt import ShotSet, Shot
import os
import json
import tensorflow as tf

if __name__ == '__main__':
    #%%
    # file path
    xfm_dir = '/mypool/DANN/data/xfm/jtext/train'
    dataset_dir = '../../file_repo/data_xfm/'
    tf.data.experimental.save(make_ds(xfm_dir),
                              os.path.join(dataset_dir,  'train'))



