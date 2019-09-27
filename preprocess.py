#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: preprocess.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/19 22:04:21
"""
import sys
reload(sys)
sys.setdefaultencoding("gb18030")

import codecs
from utils.data_io import get_data
from sklearn.model_selection import train_test_split


def train_test_split_with_feature(data_path, gen_feature_func, test_ratio=0.2, shuffle=True, encoding="gb18030"):
    """读取数据集地址，用给定的函数生成特征，并划分出训练集、验证集
    [in]  data_path: str, 数据集地址
          gen_feature_func: func, 对每行数据生成特征的函数
          encoding: str, 文件编码
    [out] train_data_list: list[str], 训练数据列表
          test_data_list: list[str], 验证数据列表
          train_feature_list: list[feature_type], 训练数据特征列表
          test_feature_list: list[feature_type], 验证数据特征列表
    """
    data = get_data(data_path)
    data_feature_list = [gen_feature_func[x] for x in data]
    return train_test_split(data_list, data_feature_list, test_size=test_ratio, shuffle=shuffle)
