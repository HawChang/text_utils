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
    """��ȡ���ݼ���ַ���ø����ĺ������������������ֳ�ѵ��������֤��
    [in]  data_path: str, ���ݼ���ַ
          gen_feature_func: func, ��ÿ���������������ĺ���
          encoding: str, �ļ�����
    [out] train_data_list: list[str], ѵ�������б�
          test_data_list: list[str], ��֤�����б�
          train_feature_list: list[feature_type], ѵ�����������б�
          test_feature_list: list[feature_type], ��֤���������б�
    """
    data = get_data(data_path)
    data_feature_list = [gen_feature_func[x] for x in data]
    return train_test_split(data_list, data_feature_list, test_size=test_ratio, shuffle=shuffle)
