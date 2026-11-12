#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/11/12 17:25
# @Author  : zhongyu
# @Site    : 
# @File    : _stack_fail.py

'''
import numpy as np
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ClipProcessor, TrimProcessor
import json
from sklearn.model_selection import train_test_split
import warnings
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #%%
    # load shots
    train_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/train')
    test_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/test')
    val_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/val')
    temp_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/_temp')
    stack_file = FileRepo('/mypool/zy/j-text/file_repo/data_process/jtext/stack')

    repo_dict = {
        'jtext': FileRepo(r'/mypool/DANN/data/xfm/jtext/raw')
    }

    for device, file_repo in repo_dict.items():
        start_time = 0.15
        vague_time = 50
        pre_time = 20

        with open(f'./config/{device}_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        tags = config['tags']
        tags_1k = tags['basic'] + tags['sxr'] + tags['axuv']
        tags_50k = tags['50k']
        all_tags = tags_1k + tags_50k
        print('basic:{}'.format(tags['basic']))
        print('basic:{}'.format(tags['sxr']))
        print('basic:{}'.format(tags['axuv']))



        shot_set = ShotSet(temp_file)
        # temp test
        shot_filter = shot_set.shot_list
        print('shots:{}'.format(len(shot_filter)))


        # # save the shots
        # np.save('../../file_repo/shot_info/v1/train_shots.npy', train_shots)
        # np.save('../../file_repo/shot_info/v1/test_shots.npy', test_shots)
        # np.save('../../file_repo/shot_info/v1/val_shots.npy', val_shots)

        # load train test & val
        train_shots = np.load('../../file_repo/shot_info/v1/train_shots.npy', allow_pickle=True).tolist()
        test_shots = np.load('../../file_repo/shot_info/v1/test_shots.npy', allow_pickle=True).tolist()
        val_shots = np.load('../../file_repo/shot_info/v1/val_shots.npy', allow_pickle=True).tolist()

        print("stacking basic")
        shot_set = shot_set.process(StackProcessor(), input_tags=[tags['basic']], output_tags=['basic'],save_repo=stack_file)
        print("stacking axuv")
        shot_set = shot_set.process(StackProcessor(), input_tags=[tags['axuv']], output_tags=['axuv'],save_repo=stack_file)
        shot_set = shot_set.process(StackProcessor(), input_tags=[tags['sxr']], output_tags=['sxr'],
                                    save_repo=stack_file)
        # print("cut for undis phase of dis shot")
        # train_set = shot_set.process(CutProcessor(pre_time=20, is_test=False),
        #                              input_tags=['basic'],
        #                              output_tags=['basic'],
        #                              shot_filter=train_shots,
        #                              save_repo=train_file,
        #                              )
        # test_set = shot_set.process(CutProcessor(pre_time=20, is_test=True),
        #                              input_tags=['basic'],
        #                              output_tags=['basic'],
        #                              shot_filter=test_shots,
        #                              save_repo=test_file,
        #                              )
        # val_set = shot_set.process(CutProcessor(pre_time=20, is_test=False),
        #                              input_tags=['basic'],
        #                              output_tags=['basic'],
        #                              shot_filter=val_shots,
        #                              save_repo=val_file,
        #                              )
        # print('labelling (classification)')
        # train_set = train_set.process(processor=BinaryLabelProcessor(is_test=False),
        #                                       input_tags=['basic'],
        #                                       output_tags=['label'],
        #                                       shot_filter=train_shots,
        #                                       save_repo=train_file,
        #                                       processes=10)
        # val_set = val_set.process(processor=BinaryLabelProcessor(is_test=False),
        #                                   input_tags=['basic'],
        #                                   output_tags=['label'],
        #                                   shot_filter=val_shots,
        #                                   save_repo=val_file,
        #                                   processes=10)
        # test_set = test_set.process(processor=BinaryLabelProcessor(is_test=True),
        #                                     input_tags=['basic'],
        #                                     output_tags=['label'],
        #                                     shot_filter=test_shots,
        #                                     save_repo=test_file,
        #                                     processes=10)
        #
        # print('save as train, test, val')
        #
        # train_set = train_set.remove_signal(tags=all_tags, keep=True, shot_filter=train_shots,
        #                                             save_repo=train_file)
        # test_set = test_set.remove_signal(tags=all_tags, keep=True, shot_filter=test_shots,
        #                                    save_repo=test_file)
        # val_set = val_set.remove_signal(tags=all_tags, keep=True, shot_filter=val_shots,
        #                                    save_repo=val_file)