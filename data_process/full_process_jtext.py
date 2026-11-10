#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/11/9 19:26
# @Author  : zhongyu
# @Site    : 
# @File    : full_process_jtext.py

'''
import numpy as np
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ClipProcessor, TrimProcessor
from processor import *
import json

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #%%
    # load shots
    train_shots = np.load('../../file_repo/shot_info/v1/train_shots.npy', allow_pickle=True).tolist()
    val_shots = np.load('../../file_repo/shot_info/v1/test_shots.npy', allow_pickle=True).tolist()
    test_shots = np.load('../../file_repo/shot_info/v1/val_shots.npy', allow_pickle=True).tolist()
    train_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/train')
    test_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/test')
    val_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/val')

    repo_dict = {
        'jtext': FileRepo(r'/mypool/DANN/data/xfm/jtext/raw')
    }

    for device, file_repo in repo_dict.items():
        start_time = 0.15
        vague_time = 50
        pre_time = 20

        with open(f'./config/{device}_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        shot_filter = train_shots + val_shots + test_shots

        tags = config['tags']
        tags_1k = tags['basic'] + tags['sxr'] + tags['axuv']
        tags_50k = tags['50k']
        all_tags = tags_1k + tags_50k

        shot_set = ShotSet(file_repo)
        # temp test
        shots_in_set = shot_set.shot_list
        missed_shots = [shot for shot in shot_filter if shot not in shots_in_set]
        print('missed shots:{}'.format(len(missed_shots)))
        print('train_shots:{}'.format(len(train_shots)))
        print('shot_filter:{}'.format(len(shot_filter)))
        print('shots_in_set:{}'.format(len(shots_in_set)))


        # print("clipping")
        # shot_set = shot_set.process(ClipProcessor(start_time, end_time_label='DownTime'), input_tags=all_tags, output_tags=all_tags,
        #                             shot_filter=shot_filter, save_repo=FileRepo(f'/mypool/zy/j-text/file_repo/data_process/{device}/trimmed'))
        # shot_set = shot_set.remove_signal(all_tags, keep=True,shot_filter=shot_filter)
        # print("trimming 1k")
        # shot_set = shot_set.process(TrimProcessor(), input_tags=[tags_1k], output_tags=[tags_1k])
        # print('trimming 50k')
        # shot_set = shot_set.process(TrimProcessor(), input_tags=[tags_50k], output_tags=[tags_50k])
        # print("slicing 50k")
        # shot_set = shot_set.process(SliceProcessor(100, 0.5), input_tags=tags_50k, output_tags=tags_50k,
        #                             save_repo=FileRepo(f'/mypool/zy/j-text/file_repo/data_process/{device}/sliced'))
        # print("re clipping")
        # shot_set = shot_set.process(ClipProcessor(start_time+0.01), input_tags=all_tags, output_tags=all_tags).\
        #     process(TrimProcessor(), input_tags=[all_tags], output_tags=[all_tags])
        # print("stacking basic")
        # shot_set = shot_set.process(StackProcessor(), input_tags=[tags['basic']], output_tags=['basic']).remove_signal(tags['basic'])
        # print("stacking axuv")
        # shot_set = shot_set.process(StackProcessor(), input_tags=[tags['axuv']], output_tags=['axuv']).remove_signal(tags['axuv'])
        # print("stacking sxr")
        # shot_set = shot_set.process(StackProcessor(), input_tags=[tags['sxr']], output_tags=['sxr']).remove_signal(tags['sxr'])
        # print('labelling (classification)')
        # shot_set = shot_set.process(ClassificationLabelProcessor(pre_time=pre_time, vague_time=vague_time), shot_filter=shot_filter,
        #                             input_tags=['basic'], output_tags=['label_1'], save_repo=FileRepo(f'/mypool/zy/j-text/file_repo/data_process/{device}/labelled'))
        #
        # train_set = shot_set.remove_signal(tags=all_tags, keep=True, shot_filter=train_shots,
        #                                             save_repo=train_file)
        # test_set = shot_set.remove_signal(tags=all_tags, keep=True, shot_filter=test_shots,
        #                                    save_repo=test_file)
        # val_set = shot_set.remove_signal(tags=all_tags, keep=True, shot_filter=val_shots,
        #                                    save_repo=val_file)


