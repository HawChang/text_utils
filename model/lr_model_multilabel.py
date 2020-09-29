#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model_multilabel.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2020/06/03 20:04:51
"""

import codecs
import logging
import math
import os
import sys
import time
from collections import defaultdict
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from utils.data_io import read_from_file
from utils.data_io import dump_pkl
from utils.data_io import load_pkl
from utils.softmax import softmax

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifierCV


class LRModelMultiLabel(object):
    def __init__(self):
        """LR多标签模型初始化
        """
        self.model_loaded = False

    def train(self,
            train_data,
            train_label):
        """训练多标签模型
        [in] train_data: array-like, 训练数据
             train_label: array-like, 训练数据的标签
        """
        logging.info("train begin")
        start_time = time.time()

        self.model = MultiOutputClassifier(RidgeClassifierCV(), n_jobs=30).fit(train_data, train_label)
        for x in self.model.predict(train_data[-2:]):
            print(x)

        logging.info("cost time %.4fs." % (time.time() - start_time))
        logging.info("train end")
        self.model_loaded = True

    def load_model(self, model_path):
        """加载模型
        [in]  model_path: str, 模型文件保存地址
        """
        logging.debug("load model begin")
        start_time = time.time()
        self.model = load_pkl(model_path)
        logging.debug("cost time %.4fs." % (time.time() - start_time))
        logging.debug("load model end")
        self.model_loaded = True

    def save(self, model_path, overwrite=True):
        """根据liblinear生成的模型文件和特征保留文件生成线上需要的multiclass特征权重文件
        [in]  model_path: str, 模型文件保存地址
              overwrite: bool, true则覆盖已有文件
        """

        if not self.model_loaded:
            raise ValueError("model should be trained or loaded before be saved.")

        logging.debug("save model begin")
        start_time = time.time()
        dump_pkl(self.model, model_path, overwrite)
        logging.debug("cost time %.4fs." % (time.time() - start_time))
        logging.debug("save model end")

    def check(self, feature_vec):
        """根据特征列表预测结果
        [in]  feature_vec : array-like, 特征列表
        [out] pred_list: list[(str, str)], 预测结果二元组列表, (类别, 置信度) 由大到小
        """
        if not self.model_loaded:
            raise ValueError("model should be loaded before check.")
        
        return self.model.predict(feature_vec)


if __name__ == "__main__":
    pass
