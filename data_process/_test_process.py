#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/19 10:55
# @Author  : zhongyu
# @Site    : 
# @File    : _test_process.py.py

'''
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor
from sklearn.model_selection import train_test_split
from util.processors import SliceProcessor
import json


if __name__ == '__main__':
    #%%
    # load file repo
    raw_file_repo = FileRepo("/mypool/J-TEXT/$shot_2$00/")
    test_process_file_repo = FileRepo("..//..//file_repo//data_process//test_process//$shot_2$00//")
    file_repo_1k = FileRepo("..//..//file_repo//data_process//1k//$shot_2$00//")
    file_repo_slice = FileRepo("..//..//file_repo//data_process//slice//$shot_2$00//")
    file_repo_raw = FileRepo("..//..//file_repo//data_process//raw//$shot_2$00//")
    shot_list = raw_file_repo.get_all_shots()
    raw_shotset = ShotSet(raw_file_repo, shot_list)
    # load tags
    file_path_tags = '../../file_repo/shot_info/v1/target_tags.json'
    with open(file_path_tags, 'r') as f:
        tag_list = json.load(f)
    basic_tags = ["\\ip", "\\bt", "\\dx"]
    tags_array_axuv_cb = find_tags('\\AXUV_CB', tag_list)
    tags_array_sxr = find_tags('\\sxr_cb', tag_list)
    tags_array_ne = find_tags('\\polaris_den', tag_list)
    tags_array_ma_tor = find_tags('\\MA_TOR1', tag_list)
    tags_array_exsad = find_tags('\\exsad', tag_list)

    #%%
    # 1.reserve target tags only
    processed_shotset = raw_shotset.remove_signal(tags=tag_list, keep=True,
                                                     save_repo=test_process_file_repo)

    # 2.
    # resample
    downsample_shotset = processed_shotset.process(processor=ResamplingProcessor(1000),
                                                  input_tags=tag_list,
                                                  output_tags=tag_list,
                                                  save_repo=file_repo_1k,
                                                  processes=10)
    print('1st processed shots:{}'.format(len(processed_shotset.shot_list)))

    # slice
    #
    slice_shotset = processed_shotset.process(processor=SliceProcessor(100, end_time=1),
                                                input_tags=tag_list,
                                                output_tags=tag_list,
                                                save_repo=file_repo_slice,
                                                processes=10)

