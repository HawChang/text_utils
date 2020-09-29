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
import logging

import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import time
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

from utils.data_io import write_to_file


class FeatureSelector(object):
    def __init__(self,
            feature_keep_percent=90,
            feature_keep_num=10,
            is_percent=True):
        """��ʼ������ѡ����
        [in]  feature_keep_percent: float, ������������
              feature_keep_num: int, ����������
              is_percent: bool, True�򰴱�������������������Ŀ
        """
        if is_percent:
            self._feature_selector = SelectPercentile(chi2, percentile=feature_keep_percent)
        else:
            self._feature_selector = SelectKBest(chi2, k=feature_keep_num)
        
        select_conf = "select top "
        select_conf += "%d%%" % feature_keep_percent if is_percent else "%d" % feature_keep_num
        logging.info("select conf: %s." % select_conf)

    def fit(self, feature_vec, label_vec, feature_name_vec, reserved_feature_file=None):
        """��������ѡ����������
        [in]  feature_vec: np.ndarray, ��������
              label_vec: np.ndarray, ��ǩ����
              feature_name_vec: list[str], ��Ӧindex����������
              reserved_feature_file: str, ����������Ϣ�����ַ
        [out] reserved_feature_name: list[str], ������������Ϣ����
        """
        start_time = time.time()
        self._feature_selector.fit(feature_vec, label_vec)
        logging.info("feature select, cost time %.4fs" % (time.time() - start_time))
        
        reserved_mask = self._feature_selector.get_support(indices=False)
        if not isinstance(feature_name_vec, np.ndarray):
            feature_name_vec = np.array(feature_name_vec)
        #logging.info("feature_name_vec shape : %s" % str(feature_name_vec.shape))
        reserved_feature_name = feature_name_vec[reserved_mask]
        #logging.info("score shape : %s" % str(self._feature_selector.scores_.shape))
        reserved_feature_score = self._feature_selector.scores_[reserved_mask]
        logging.info("feature origin num %d, reserved num %d" % \
                (feature_vec.shape[1], len(reserved_feature_name)))
        
        if reserved_feature_file is not None:
            #�б�������������������桢�÷���Ϣ
            reserved_feature_list = zip(reserved_feature_name, reserved_feature_score)

            reserved_feature_list = sorted(reserved_feature_list, key=lambda x:x[1], reverse=True)
            reserved_feature_name = [x[0] for x in reserved_feature_list]

            write_to_file(enumerate(reserved_feature_list), reserved_feature_file, \
                    write_func=lambda x : "%d\t%s\t%.4f" % (x[0] + 1, x[1][0], x[1][1]))
        return reserved_feature_name

    def transform(self, feature_vec):
        """��������ֻ����fit�׶�ѡ�е�����
        [in]  feature_vec: np.ndarrayr, ��������
        [out] reserved_feature_vec: np.ndarray, ֻʣ�±�����������������
        """
        return self._feature_selector.transform(feature_vec)
