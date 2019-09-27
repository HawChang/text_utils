#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_io.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/19 20:44:30
"""

import sys
reload(sys)
sys.setdefaultencoding("gb18030")

import os
import codecs
import pickle

from logger import Logger

log = Logger().get_logger()


def get_data(data_path, verbose=False):
    """获取该文件(或目录下所有文件)的数据
    [in]  data_path : str, 数据集地址
          verbose   : bool, 是否展示处理信息
    [out] data : list[str], 该文件(或目录下所有文件)的数据
    """
    file_list = get_file_name_list(data_path, verbose)
    data = list()
    for file_path in file_list:
        data.extend(read_from_file(file_path))
    return data


def get_file_name_list(data_path, verbose=False):
    """生成构成数据集的文件列表
        如果数据集地址是文件，则返回列表中只有该文件地址
        如果数据集地址是目录，则返回列表中包括该目录下所有文件名称(忽略'.'开头的文件)
    [in]  data_path : str, 数据集地址
          verbose   : bool, 是否展示处理信息
    [out] file_list : list[str], 数据集文件名称列表
    """
    file_list = list()
    log.info("check data path: %s." % data_path)
    # 首先检查原始数据是文件还是文件夹
    if os.path.isdir(data_path):
        log.info("data path is directory.")
        files = os.listdir(data_path)
        # 遍历文件夹中的每一个文件
        for file_name in files:
            # 如果文件名以.开头 说明该文件隐藏 不是正常的数据文件
            if len(file_name) == 0 or file_name[0] == ".":
                continue
            file_path = os.path.join(data_path, file_name)
            file_list.append(file_path)
    elif os.path.isfile(data_path):
        log.info("data path is file.")
        file_list.append(data_path)
    else:
        raise TypeError("unknown type of data path : %s." % data_path)

    log.info("file list:")
    for index, file_name in enumerate(file_list):
        log.info("#%d: %s" % (index + 1, file_name))
    return file_list


def read_from_file(file_path, read_func=lambda x:x, encoding="gb18030"):
    """加载文件中的词
    [in] file_path: str, 文件地址
    [out] word_list: list[str], 单词列表
    """
    word_list = list()
    with codecs.open(file_path, "r", encoding) as rf:
        for line in rf:
            word_list.append(read_func(line.strip("\n")))
    return word_list


def write_to_file(text_list, dst_file_path, write_func=lambda x:x, encoding="gb18030"):
    """将文本列表存入目的文件地址
    [in]  text_list: list[str], 文本列表
          dst_file_path: str, 目的文件地址
    """
    with codecs.open(dst_file_path, "w", encoding) as wf:
        wf.write("\n".join([write_func(x) for x in text_list]))


def load_pkl(pkl_path):
    """加载对象
    [in]  pkl_path: str, 对象文件地址
    [out] obj: class, 对象
    """
    with open(pkl_path, 'rb') as rf:
        return pickle.load(rf)


def dump_pkl(obj, pkl_path, overwrite=False):
    """存储对象
    [in]  obj: class, 对象
          pkl_path: str, 对象文件地址
          overwrite: bool, 是否覆盖，False则当文件存在时不存储
    """
    if len(pkl_path) == 0 or pkl_path is None:
        log.warning("pkl_path(\"%s\") illegal." % pkl_path)
    elif os.path.exists(pkl_path) and not overwrite:
        log.warning("pkl_path(\"%s\") already exist and not over write." % pkl_path)
    else:
        with open(pkl_path, 'wb') as wf:
            pickle.dump(obj, wf)
        log.debug("save to \"%s\" succeed." % pkl_path)
