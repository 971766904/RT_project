#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/18 16:09
# @Author  : zhongyu
# @Site    : 
# @File    : _check_tags_in_shots.py.py

'''
from jddb.file_repo import FileRepo
import numpy as np
from util.basic_processor import find_tags
from sklearn.preprocessing import StandardScaler
import json

if __name__ == '__main__':
    #%%
    jtext_file_repo = FileRepo("/mypool/J-TEXT/$shot_2$00/")
    file_path_shots = '../../file_repo/shot_info/v1/shot_info.json'
    file_path_tags = '../../file_repo/shot_info/v1/target_tags.json'
    # load shots and tags
    with open(file_path_shots, 'r') as f:
        shot_list = json.load(f)
    with open(file_path_tags, 'r') as f:
        tag_list = json.load(f)

    tags_array_ma_polA = find_tags('\\MA_POLA', tag_list)
    tags_array_ma_polB = find_tags('\\MA_POLB', tag_list)

    #%%
    # check tags in shots
    for shot in shot_list:
        shot_tags = jtext_file_repo.get_tag_list(shot)
        # check if tags_array_ma_polA and tags_array_ma_polB are in shot_tags
        for tag in tags_array_ma_polA + tags_array_ma_polB:
            if tag not in shot_tags:
                print(f"Shot {shot} is missing tag {tag}")




