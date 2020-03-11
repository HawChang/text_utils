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

import os
import sys
import time

from sklearn.model_selection import GridSearchCV

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from utils.logger import Logger

log = Logger().get_logger()

def grid_search_cv(estimator, params, train_data, test_data, scoring=None,
        cv=5, n_jobs=-1, refit=True):
    """网格搜索优化参数
          在使用gridsearchcv时 SpectralClustering没有predict函数
          因为其继承的ClusterMixin中没有predict函数 只有fit_predict函数
          因此将ClusterMixin中的fit_predict改为了predict
    [in]  estimator: 模型
          params: dict, 需要搜索的各参数及其范围
          train_data: array-like shape(n_samples, n_features), 训练数据
          test_data: array-like shape(n_samples, n_features), 验证数据
          scoring: method, 得分评估函数，若为None则使用模型自带的score函数
          cv: int, 交叉检验参数
          n_jobs: int, 并发数量, -1表示尽可能多
          refit: bool, true则会用得分最高的参数重新训练得到最终模型
    [out] gsc: GridSearchCV, 网格搜索类
    """

    gsc = GridSearchCV(estimator, params, cv=cv, n_jobs=n_jobs, refit=refit, scoring=scoring)

    log.info("start GridSearchCV")
    start_time = time.time()
    # 评估聚类效果依赖train_data和模型聚类结果
    gsc.fit(train_data, train_data)
    log.debug("cost time : %.4fs" % (time.time() - start_time))
    grid_search_cv_res(gsc)
    log.info("best params : %s" % str(gsc.best_params_))
    return gsc

def grid_search_cv_res(gsc):
    """展示GridSearchCV的结果
    [in]  gsc: GridSearchCV, 网格搜索类, 包含各参数组合的评估信息
    """
    means = gsc.cv_results_['mean_test_score']
    stds = gsc.cv_results_['std_test_score']
    params_list = gsc.cv_results_['params']
    log.info("GridSearchCV score info:")
    for mean, std, params in zip(means, stds, params_list):
        log.debug("\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
