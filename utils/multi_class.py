#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: multi_class.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/04 20:13:00
"""

import codecs
import math
from collections import defaultdict
from logger import Logger

log = Logger().get_logger()

def load_lr_model_feature_weight(file_name, encoding="gb18030"):
    """
    [in]  file_name: str, 模型文件名
    [out] feature_weight_dict: dict, str(label) -> list[float]
    """
    label_list = None
    feature_weight_dict = defaultdict(list)
    with codecs.open(file_name, "r" ,encoding) as rf:
        for index, line in enumerate(rf):
            parts = line.strip("\n").split("\t")
            assert len(parts) == 2, \
                    "wrong field num at line #%d. Expect 2, actual %d" % (index + 1, len(parts))
            if index == 0 and line.startswith("classes:"): 
                label_list = parts[-1].split(",")
                continue
            feature_name = parts[0]
            feature_weights = parts[1].split(" ")
            if label_list is None:
                # 说明第一行就是权值 而没有label介绍
                # 则默认以列数下标作为类别
                label_list = range(len(feature_weights))

            if len(label_list) > 2:
                # 当类别数大于2时 每个特征对每类都有一个权值
                assert len(feature_weights) == len(label_list), \
                        "feature_weight_col_num(%d) != label_num(%d)." % (len(feature_weights), len(label_list))
                feature_weight_dict[feature_name] = [float(x) for x in feature_weights]
            else:
                feature_weight_dict[feature_name] = [float(feature_weights[0]), -float(feature_weights[0])]
    return label_list, feature_weight_dict

def lr_predict(features, feature_weight_dict, label_list, min_conf=0.05):
    """计算多分类预测结果
    [in]  features : 特征列表
          feature_weight_dict: 特征权值字典
          label_list: 预测结果
    [out] pred_list: 预测结果
    """
    label_value = defaultdict(lambda: 0.0)
    total = 0.0
    for feature in features:
        if feature not in feature_weight_dict:
            continue
        for index, weight in enumerate(feature_weight_dict[feature]):
            try:
                label_value[label_list[index]] += weight
            except IndexError as e:
                print("index error. index = %d." % index)
                raise e

    #log.debug("label_weght_sum: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value) for label,value in label_value.items() ]))

    for label, sum_weight in label_value.items():
        label_value[label] = 1.0 / (1.0 + math.exp(-sum_weight))
        total += label_value[label]

    #log.debug("label_value: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value) for label,value in label_value.items() ]))

    pred_list = list()
    for label, weight_pred in sorted(label_value.items(), key = lambda x: x[1], reverse = True):
        pred_proba = weight_pred / total
        if pred_proba < min_conf:
            continue
        pred_list.append((label, "%.4f" % pred_proba))
    return pred_list
