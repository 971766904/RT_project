#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/16 15:38
# @Author  : zhongyu
# @Site    : 
# @File    : data_process.py.py

'''
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor
from sklearn.model_selection import train_test_split
import json

if __name__ == '__main__':
    #%%
    # load file repo
    raw_file_repo = FileRepo("/mypool/J-TEXT/$shot_2$00/")
    process_file_repo = FileRepo("..//..//file_repo//data_process//temp//$shot_2$00//")
    train_file = FileRepo("..//..//file_repo//data_process//train//$shot_2$00//")
    val_file = FileRepo("..//..//file_repo//data_process//val//$shot_2$00//")
    test_file = FileRepo("..//..//file_repo//data_process//test//$shot_2$00//")
    #load shots and tags from json
    file_path_shots = '../../file_repo/shot_info/v1/shot_info.json'
    file_path_tags = '../../file_repo/shot_info/v1/target_tags.json'
    with open(file_path_shots, 'r') as f:
        shot_list = json.load(f)
    with open(file_path_tags, 'r') as f:
        tag_list = json.load(f)
    basic_tags = ["\\ip", "\\bt", "\\dx"]
    tags_array_axuv_cb = find_tags('\\AXUV_CB', tag_list)
    tags_array_sxr = find_tags('\\sxr_cb', tag_list)
    tags_array_ne = find_tags('\\polaris_den', tag_list)
    tags_array_ma_tor = find_tags('\\MA_TOR1', tag_list)
    tags_array_exsad = find_tags('\\exsad', tag_list)

    #
    is_disrupt = []
    for shot in shot_list:
        dis_label = raw_file_repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])
    print('all shots:{}'.format(len(shot_list)))
    print('disruption shots:{}'.format(sum(is_disrupt)))

    # %% build model specific data
    # train test split on shot not sample according to whether shots are disruption
    # set test_size=0.5 to get 50% shots as test set
    train_shots, test_shots, train_label, _ = \
        train_test_split(shot_list, is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=is_disrupt)
    train_shots, val_shots, _, _ = \
        train_test_split(train_shots, train_label, test_size=0.2,
                         random_state=1, shuffle=True, stratify=train_label)
    raw_shotset = ShotSet(raw_file_repo, shot_list)

    # %%
    # 1.reserve target tags only
    processed_shotset = raw_shotset.remove_signal(tags=tag_list, keep=True,
                                                     save_repo=process_file_repo)


