#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   process_data.py
Author:   zhanghao55@baidu.com
Date  :   21/01/08 10:37:57
Desc  :   
"""

import os
import logging
import pandas as pd

from text_utils.utils.data_io import get_file_name_list, write_to_file


def process_origin_poetry(data_dir, dst_dir, overwrite=False, verbose=False):
    if not os.path.isdir(dst_dir):
        logging.debug("create data dir: {}".format(dst_dir))
        os.mkdir(dst_dir)

    def gen_poet_iter(data_path):
        df = pd.read_csv(data_path)
        for index, row in df.iterrows():
            try:
                title = row[0].replace("\t", "\\t")
                dynasty = row[1].replace("\t", "\\t")
                author = row[2].replace("\t", "\\t")
                poet = row[3].replace("\t", "\\t")
                yield "\t".join([title, dynasty, author, poet])
            except AttributeError as e:
                logging.warning("parse poet fail at line #{}".format(index + 1))

    for data_path in get_file_name_list(data_dir):
        file_name = data_path[data_path.rfind("/")+1:]
        # 隐藏文件 或 不为csv的文件 跳过
        if file_name.startswith(".") or (not file_name.endswith("csv")):
            continue
        if verbose:
            logging.info("process: {}".format(data_path))
        dst_path = os.path.join(dst_dir, file_name)
        if os.path.exists(dst_path):
            if not overwrite:
                if verbose:
                    logging.info("{} already processed, skip.".format(file_name))
                continue
        write_to_file(gen_poet_iter(data_path), dst_path)


if __name__ == "__main__":
    pass
