#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2026/11/13 01:14
# @Author  : zhongyu
# @Site    : 
# @File    : debug_stack.py.py

'''

import os
import sys
import json
import inspect
import traceback
from pathlib import Path

import numpy as np
import jddb

from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from util.basic_processor import StackProcessor


DATA_DIR = Path(
    "/mypool/zy/j-text/file_repo/data_process/jtext"
)

INPUT_DIR = DATA_DIR / "_temp"
OUTPUT_DIR = DATA_DIR / "_stack_debug"


def show_environment():
    print("\n========== Environment ==========", flush=True)
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("JDDB path:", jddb.__file__)
    print("JDDB version:", getattr(jddb, "__version__", "unknown"))

    print("Working directory:", os.getcwd())
    print("Input exists:", INPUT_DIR.exists())
    print("Output exists:", OUTPUT_DIR.exists())

    print("\n========== JDDB Process ==========", flush=True)

    print("ShotSet.process signature:")
    print(inspect.signature(ShotSet.process))

    print("ShotSet.process source:")
    try:
        print(inspect.getsource(ShotSet.process))
    except Exception as e:
        print("Cannot read source:", repr(e))


def main():
    show_environment()

    config_path = Path("./config/jtext_config.json")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    basic_tags = config["tags"]["basic"]

    print("\n========== Tags ==========", flush=True)
    print("Basic tags count:", len(basic_tags))
    print("Basic tags:", basic_tags)

    print("\n========== FileRepo ==========", flush=True)

    input_repo = FileRepo(str(INPUT_DIR))
    output_repo = FileRepo(str(OUTPUT_DIR))

    shot_set = ShotSet(input_repo)
    shot_list = list(shot_set.shot_list)

    print("Input shot count:", len(shot_list))
    print("First 10 shots:", shot_list[:10])

    if not shot_list:
        raise RuntimeError("Input ShotSet contains no shots")

    test_shot = shot_list[0]

    print("\n========== Processing ==========", flush=True)
    print("Test shot:", test_shot)
    print("Processes: 1")

    processor = StackProcessor()

    try:
        result = shot_set.process(
            processor,
            input_tags=[basic_tags],
            output_tags=["basic"],
            shot_filter=[test_shot],
            save_repo=output_repo,
            processes=1
        )

        print("\nProcess returned successfully")
        print("Result type:", type(result))

        if result is not None:
            print(
                "Result shots:",
                getattr(result, "shot_list", "unavailable")
            )

    except Exception:
        print("\nPROCESS EXCEPTION:")
        traceback.print_exc()
        raise

    print("\n========== Output ==========", flush=True)

    print("Output path:", OUTPUT_DIR)
    print("Output exists:", OUTPUT_DIR.exists())

    if OUTPUT_DIR.exists():
        files = list(OUTPUT_DIR.rglob("*"))
        print("Output entries:", len(files))
        for path in files[:20]:
            print(path)

    saved_set = ShotSet(output_repo)
    print("Saved shot list:", saved_set.shot_list)


if __name__ == "__main__":
    main()
