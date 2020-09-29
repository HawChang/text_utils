#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: parameter_optimizer.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/18 17:35:01
"""

import logging
import os
import sys
import time

from sklearn.model_selection import GridSearchCV

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

def grid_search_cv(estimator, params, train_data, test_data, scoring=None,
        cv=5, n_jobs=-1, refit=True):
    """���������Ż�����
          ��ʹ��gridsearchcvʱ SpectralClusteringû��predict����
          ��Ϊ��̳е�ClusterMixin��û��predict���� ֻ��fit_predict����
          ��˽�ClusterMixin�е�fit_predict��Ϊ��predict
    [in]  estimator: ģ��
          params: dict, ��Ҫ�����ĸ��������䷶Χ
          train_data: array-like shape(n_samples, n_features), ѵ������
          test_data: array-like shape(n_samples, n_features), ��֤����
          scoring: method, �÷�������������ΪNone��ʹ��ģ���Դ���score����
          cv: int, ����������
          n_jobs: int, ��������, -1��ʾ�����ܶ�
          refit: bool, true����õ÷���ߵĲ�������ѵ���õ�����ģ��
    [out] gsc: GridSearchCV, ����������
    """

    gsc = GridSearchCV(estimator, params, cv=cv, n_jobs=n_jobs, refit=refit, scoring=scoring)

    logging.info("start GridSearchCV")
    start_time = time.time()
    # ��������Ч������train_data��ģ�;�����
    gsc.fit(train_data, train_data)
    logging.debug("cost time : %.4fs" % (time.time() - start_time))
    grid_search_cv_res(gsc)
    logging.info("best params : %s" % str(gsc.best_params_))
    return gsc

def grid_search_cv_res(gsc):
    """չʾGridSearchCV�Ľ��
    [in]  gsc: GridSearchCV, ����������, ������������ϵ�������Ϣ
    """
    means = gsc.cv_results_['mean_test_score']
    stds = gsc.cv_results_['std_test_score']
    params_list = gsc.cv_results_['params']
    logging.info("GridSearchCV score info:")
    for mean, std, params in zip(means, stds, params_list):
        logging.debug("\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
