#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/23 14:03:51
"""

import sys
reload(sys)
sys.setdefaultencoding("gb18030")

import time
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from utils.logger import Logger

log = Logger().get_logger()


def train_lr_model(feature_vec, label_vec, **lr_parameters):
    """训练lrmodel
    """
    hyper_parameters = {"C": [0.1, 0.2, 0.5, 1, 2, 5, 10]}
    start_time = time.time()
    log.info("train model.")
    lr_model = LogisticRegression(
            solver="liblinear",
            fit_intercept=False,
            **lr_parameters)
    search_res = GridSearchCV(lr_model, hyper_parameters, scoring="accuracy", cv=5, refit=True)\
            .fit(feature_vec, label_vec)
    log.info("cost time %.4fs, best score = %.4f, parameters = %s" % \
            (time.time() - start_time, search_res.best_score_, search_res.best_params_))
    return search_res.best_estimator_


if __name__ == "__main__":
    from utils.data_io import load_pkl
    from utils.data_io import dump_pkl
    #train_feature_vec = load_pkl("local_data/train_feature_vec.pkl")
    #train_label_vec = load_pkl("local_data/train_label_vec.pkl")
    test_feature_vec = load_pkl("local_data/test_feature_vec.pkl")
    test_label_vec = load_pkl("local_data/test_label_vec.pkl")

    print("train shape = %s " % str(train_feature_vec.shape))

    #lr_model = train_lr_model(train_feature_vec, train_label_vec)
    #dump_pkl(lr_model, "model/lr_model.pkl", overwrite=True)
    lr_model = load_pkl("model/lr_model.pkl")
