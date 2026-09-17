#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/9/11 15:46
# @Author  : zhongyu
# @Site    : 
# @File    : get_shots&tags.py

'''
from jddb.meta_db import MetaDB
from jddb.file_repo import FileRepo
from jddb.processor import Shot
import json
from util.basic_processor import find_tags


# %% connect to the MetaDB
# from pymongo import MongoClient
#
# client = MongoClient(
#     host="127.0.0.1",
#     port=18107,
#     username="jtext103",
#     password="jtext103.",
#     authSource="admin",
#     serverSelectionTimeoutMS=5000
# )
#
# db = client["DDB"]
#
# print(client.admin.command("ping"))
# print("MongoDB 连接成功")

connection_str = {
    "host": "127.0.0.1",
    "port": 18107,
    "username": "jtext103",
    "password": "jtext103.",
    "database": "admin"

}
collection = "Labels"

db = MetaDB(connection_str, collection)


if __name__ == '__main__':
    # read json file
    file_path_json = "../../file_repo/shot_info/shot_collect/hdf5_with_all_tags.json"
    with open(file_path_json, "r") as f:
        shot_info = json.load(f)


    shot_list = [shot for shot in range(1111987, 1114423)]
    complete_disruption_shots = db.query_valid(
        shot_list=shot_info,
        label_true=[ "ip", "bt"]
    )
    print(complete_disruption_shots)
    print(len(complete_disruption_shots))

    #%%
    # get shot list and tag list
    file_path_shot_list = "../../file_repo/shot_info/shot_collect/all_shots.json"
    file_path_tag_list = "../../file_repo/shot_info/shot_collect/all_tags.json"
    with open(file_path_shot_list, "r") as f:
        all_shots = json.load(f)
    with open(file_path_tag_list, "r") as f:
        all_tags = json.load(f)

    #%%
    # read example shot data
    example_file_repo = FileRepo("../../file_repo/example/_temp_data/")
    shots = example_file_repo.get_all_shots()
    tag_list = example_file_repo.get_tag_list(shots[0])
    tags_array_axuv_cb = find_tags('\\AXUV_CB', tag_list)
    tags_array_sxr = find_tags('\\sxr_cb', tag_list)
    tags_array_ne = find_tags('\\polaris_den', tag_list)
    tags_array_ma_tor = find_tags('\\MA_TOR1', tag_list)
    tags_array_exsad = find_tags('\\exsad', tag_list)
    tags_array_ma_polA = find_tags('\\MA_POLA', tag_list)
    tags_array_ma_polB = find_tags('\\MA_POLB', tag_list)
    basic_tags = [  "\\ip",  "\\bt",  "\\dx"]
    target_tags = (basic_tags + tags_array_axuv_cb + tags_array_sxr + tags_array_ne + tags_array_ma_tor +
                   tags_array_exsad + tags_array_ma_polA + tags_array_ma_polB)
    # save the target tags to json file
    target_tags_file_path = "../../file_repo/shot_info/v1/target_tags.json"
    with open(target_tags_file_path, "w") as f:
        json.dump(target_tags, f, indent=4)
    print(f"Target tags have been saved to {target_tags_file_path}")
    #save the shot_info to json file
    shot_info_file_path = "../../file_repo/shot_info/v1/shot_info.json"
    with open(shot_info_file_path, "w") as f:
        json.dump(shot_info, f, indent=4)
    print(f"Shot info has been saved to {shot_info_file_path}")







