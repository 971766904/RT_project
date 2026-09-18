#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/17 16:38
# @Author  : zhongyu
# @Site    : 
# @File    : mean_std_calculate.py.py

'''
from jddb.file_repo import FileRepo
import numpy as np
from util.basic_processor import find_tags
from sklearn.preprocessing import StandardScaler
import json

if __name__ == '__main__':
    #%%
    jtext_file_repo = FileRepo("..//..//file_repo//jtext//$shot_2$00//")
    file_path_shots = '../../file_repo/shot_info/v1/shot_info.json'
    file_path_tags = '../../file_repo/shot_info/v1/target_tags.json'
    # load shots and tags
    with open(file_path_shots, 'r') as f:
        shot_list = json.load(f)
    with open(file_path_tags, 'r') as f:
        tag_list = json.load(f)
    tags_array_axuv_cb = find_tags('\\AXUV_CB', tag_list)
    tags_array_sxr = find_tags('\\sxr_cb', tag_list)
    tags_array_ne = find_tags('\\polaris_den', tag_list)
    tags_array_ma_tor = find_tags('\\MA_TOR1', tag_list)
    tags_array_exsad = find_tags('\\exsad', tag_list)
    # tags_array_ma_polA = find_tags('\\MA_POLA', tag_list)
    # tags_array_ma_polB = find_tags('\\MA_POLB', tag_list)
    basic_tags = [  "\\ip",  "\\bt",  "\\dx"]

    #%%
    # initialization
    normalization_dic = dict()
    data_cal = dict()
    data_cal['\\AXUV_CB'] = np.empty(0)
    data_cal['\\sxr_cb'] = np.empty(0)
    data_cal['\\polaris_den'] = np.empty(0)
    data_cal['\\MA_TOR1'] = np.empty(0)
    data_cal['\\exsad'] = np.empty(0)
    # data_cal['\\MA_POLA'] = np.empty(0)
    # data_cal['\\MA_POLB'] = np.empty(0)
    for tag in basic_tags:
        data_cal[tag] = np.empty(0)

    #%%
    # calculate by shot
    for shot in shot_list:
        # axuv array calculate
        axuv_array_data = jtext_file_repo.read_data(shot, tags_array_axuv_cb)
        for tag in tags_array_axuv_cb:
            data_cal['\\AXUV_CB'] = np.concatenate((data_cal['\\AXUV_CB'], axuv_array_data[tag]))
        # sxr array calculate
        sxr_array_data = jtext_file_repo.read_data(shot, tags_array_sxr)
        for tag in tags_array_sxr:
            data_cal['\\sxr_cb'] = np.concatenate((data_cal['\\sxr_cb'], sxr_array_data[tag]))
        # ne array calculate
        ne_array_data = jtext_file_repo.read_data(shot, tags_array_ne)
        for tag in tags_array_ne:
            data_cal['\\polaris_den'] = np.concatenate((data_cal['\\polaris_den'], ne_array_data[tag]))
        # ma_tor array calculate
        ma_tor_array_data = jtext_file_repo.read_data(shot, tags_array_ma_tor)
        for tag in tags_array_ma_tor:
            data_cal['\\MA_TOR1'] = np.concatenate((data_cal['\\MA_TOR1'], ma_tor_array_data[tag]))
        # exsad array calculate
        exsad_array_data = jtext_file_repo.read_data(shot, tags_array_exsad)
        for tag in tags_array_exsad:
            data_cal['\\exsad'] = np.concatenate((data_cal['\\exsad'], exsad_array_data[tag]))
        # # ma_polA array calculate
        # ma_polA_array_data = jtext_file_repo.read_data(shot, tags_array_ma_polA)
        # for tag in tags_array_ma_polA:
        #     data_cal['\\MA_POLA'] = np.concatenate((data_cal['\\MA_POLA'], ma_polA_array_data[tag]))
        # # ma_polB array calculate
        # ma_polB_array_data = jtext_file_repo.read_data(shot, tags_array_ma_polB)
        # for tag in tags_array_ma_polB:
        #     data_cal['\\MA_POLB'] = np.concatenate((data_cal['\\MA_POLB'], ma_polB_array_data[tag]))

        # basic calculate
        basic_data = jtext_file_repo.read_data(shot, basic_tags)
        for tag in basic_tags:
            data_cal[tag] = np.concatenate((data_cal[tag], basic_data[tag]))

    #%%
    axuv_scaler = StandardScaler()
    axuv_scaler.fit(data_cal['\\AXUV_CB'].reshape(-1, 1))
    normalization_dic['\\AXUV_CB'] = [axuv_scaler.mean_.tolist(), np.sqrt(axuv_scaler.var_).tolist()]
    sxr_scaler = StandardScaler()
    sxr_scaler.fit(data_cal['\\sxr_cb'].reshape(-1, 1))
    normalization_dic['\\sxr_cb'] = [sxr_scaler.mean_.tolist(), np.sqrt(sxr_scaler.var_).tolist()]
    ne_scaler = StandardScaler()
    ne_scaler.fit(data_cal['\\polaris_den'].reshape(-1, 1))
    normalization_dic['\\polaris_den'] = [ne_scaler.mean_.tolist(), np.sqrt(ne_scaler.var_).tolist()]
    ma_tor_scaler = StandardScaler()
    ma_tor_scaler.fit(data_cal['\\MA_TOR1'].reshape(-1, 1))
    normalization_dic['\\MA_TOR1'] = [ma_tor_scaler.mean_.tolist(), np.sqrt(ma_tor_scaler.var_).tolist()]
    exsad_scaler = StandardScaler()
    exsad_scaler.fit(data_cal['\\exsad'].reshape(-1, 1))
    normalization_dic['\\exsad'] = [exsad_scaler.mean_.tolist(), np.sqrt(exsad_scaler.var_).tolist()]
    # ma_polA_scaler = StandardScaler()
    # ma_polA_scaler.fit(data_cal['\\MA_POLA'].reshape(-1, 1))
    # normalization_dic['\\MA_POLA'] = [ma_polA_scaler.mean_.tolist(), np.sqrt(ma_polA_scaler.var_).tolist()]
    # ma_polB_scaler = StandardScaler()
    # ma_polB_scaler.fit(data_cal['\\MA_POLB'].reshape(-1, 1))
    # normalization_dic['\\MA_POLB'] = [ma_polB_scaler.mean_.tolist(), np.sqrt(ma_polB_scaler.var_).tolist()]
    for tag in basic_tags:
        scaler = StandardScaler()
        scaler.fit(data_cal[tag].reshape(-1, 1))
        normalization_dic[tag] = [scaler.mean_.tolist(), np.sqrt(scaler.var_).tolist()]

    #%%
    # save normalization data
    json_file_path = 'config/normalization_params.json'
    json_data = json.dumps(normalization_dic, indent=4)  # indent参数用于指定缩进空格数，使JSON文件更易读
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
    print(f'Data has been saved to {json_file_path}')