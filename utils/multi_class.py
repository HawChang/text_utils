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
    [in]  file_name: str, ģ���ļ���
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
                # ˵����һ�о���Ȩֵ ��û��label����
                # ��Ĭ���������±���Ϊ���
                label_list = range(len(feature_weights))

            if len(label_list) > 2:
                # �����������2ʱ ÿ��������ÿ�඼��һ��Ȩֵ
                assert len(feature_weights) == len(label_list), \
                        "feature_weight_col_num(%d) != label_num(%d)." % (len(feature_weights), len(label_list))
                feature_weight_dict[feature_name] = [float(x) for x in feature_weights]
            else:
                feature_weight_dict[feature_name] = [float(feature_weights[0]), -float(feature_weights[0])]
    return label_list, feature_weight_dict

def lr_predict(features, feature_weight_dict, label_list, min_conf=0.05):
    """��������Ԥ����
    [in]  features : �����б�
          feature_weight_dict: ����Ȩֵ�ֵ�
          label_list: Ԥ����
    [out] pred_list: Ԥ����
    """
    label_value = defaultdict(lambda: 0.0)
    total = 0.0
    for feature in features:
        if feature not in feature_weight_dict:
            continue
        for index, weight in enumerate(feature_weight_dict[feature]):
            label_value[label_list[index]] += weight

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
