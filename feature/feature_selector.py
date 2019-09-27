#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: feature_selector.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/20 20:13:29
"""

import sys
reload(sys)
sys.setdefaultencoding("gb18030")

import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import time
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

from utils.logger import Logger
from utils.data_io import write_to_file


log = Logger().get_logger()


class FeatureSelector(object):
    def __init__(self,
            feature_keep_percent=90,
            feature_keep_num=10,
            is_percent=True):
        """初始化特征选择函数
        [in]  feature_keep_percent: float, 特征保留比例
              feature_keep_num: int, 特征保留数
              is_percent: bool, True则按比例保留特征，否则按数目
        """
        if is_percent:
            self._feature_selector = SelectPercentile(chi2, percentile=feature_keep_percent)
        else:
            self._feature_selector = SelectKBest(chi2, k=feature_keep_num)
        select_conf = "select top "
        select_conf += "%d%%" % feature_keep_percent if is_percent else "%d" % feature_keep_num
        log.info("select conf: %s." % select_conf)

    def fit(self, feature_vec, label_vec, feature_name_vec, reserved_feature_file=None):
        """根据数据选择保留的特征
        [in]  feature_vec: np.ndarray, 特征矩阵
              label_vec: np.ndarray, 标签向量
              feature_name_vec: list[str], 对应index的特征字面
              reserved_feature_file: str, 保留特征信息储存地址
        [out] reserved_feature_name: list[str], 保留的特征信息字面
        """
        start_time = time.time()
        self._feature_selector.fit(feature_vec, label_vec)
        log.info("feature select, cost time %.4fs" % (time.time() - start_time))
        
        reserved_mask = self._feature_selector.get_support(indices=False)
        if not isinstance(feature_name_vec, np.ndarray):
            feature_name_vec = np.array(feature_name_vec)
        #log.info("feature_name_vec shape : %s" % str(feature_name_vec.shape))
        reserved_feature_name = feature_name_vec[reserved_mask]
        #log.info("score shape : %s" % str(self._feature_selector.scores_.shape))
        reserved_feature_score = self._feature_selector.scores_[reserved_mask]
        log.info("feature origin num %d, reserved num %d" % \
                (feature_vec.shape[1], len(reserved_feature_name)))
        
        if reserved_feature_file is not None:
            #列表，保存各保留特征的index、字面、得分信息
            reserved_feature_list = zip(range(len(reserved_feature_name)), \
                    reserved_feature_name, reserved_feature_score)

            reserved_feature_list = sorted(reserved_feature_list, key=lambda x:x[2], reverse=True)

            write_to_file(reserved_feature_list, reserved_feature_file, \
                    write_func=lambda x : "%d\t%s\t%.4f" % (x[0], x[1], x[2]))
        return reserved_feature_name

    def transform(self, feature_vec):
        """特征矩阵只保留fit阶段选中的特征
        [in]  feature_vec: np.ndarrayr, 特征矩阵
        [out] reserved_feature_vec: np.ndarray, 只剩下保留特征的特征矩阵
        """
        return self._feature_selector.transform(feature_vec)
