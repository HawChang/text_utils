#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: cluster.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/15 16:55:49
"""

import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import make_scorer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GridSearchCV

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from utils.parameter_optimizer import grid_search_cv


def cluster_score(X, y_pred):
    """计算聚类的效果 越大越好
    [in]  X: array-like, shape(n_samples, n_features), 数据矩阵
          y_pred: array-like, shape(n_samples,), 聚类结果
    [out] score: double, 聚类效果得分
    """
    try:
        score = calinski_harabaz_score(X.toarray(), y_pred)
        logging.info("Calinski-Harabasz Score : %.4f" % score)
    except ValueError as e:
        score = -1.0
        logging.info("Calinski-Harabasz Score Fail : %.4f" % score)
    return score


def mini_batch_kmeans(n_clusters=None, init="k-means++", batch_size=100,
        n_init=10, max_no_improvement=10, verbose=0, **kmeans_args):
    """ 批量kmeans 可调参数:n_clusters
    [in]  n_clusters: int, 聚类簇个数
          init: str, 初始化方法
          batch_size: int, 批量聚类时每批大小
          n_init: int, 重复聚类次数
          max_no_improvement: int, 指定多少次没有提升后即停止聚类
          verbose: int, 越大则展示的信息越多
          **kmeans_args: dict, MiniBatchKMeans其他参数
    [out] MiniBatchKMeans, 聚类模型
    """
    return MiniBatchKMeans(n_clusters = n_clusters, init=init, batch_size=batch_size, n_init=n_init,
            max_no_improvement=max_no_improvement, verbose=verbose, **kmeans_args)


def spectral_clustering(n_clusters=None, gamma=None, assign_labels="kmeans", **spectral_args):
    """谱聚类 可调参数:n_clusters gamma
          在使用gridsearchcv时 SpectralClustering没有predict函数
          因为其继承的ClusterMixin中没有predict函数 只有fit_predict函数
          因此将ClusterMixin中的fit_predict改为了predict
    [in]  n_clusters: int, 投影子空间的维度
          gamma: double, 核函数rbf的kernel coefficient
          assign_labels: str, 在嵌入空间分配标签的策略
          **spectral_args: dict, SpectralClustering其他参数配置
    [out] SpectralClustering, 谱聚类模型
    """
    return SpectralClustering(n_clusters = n_clusters, gamma = gamma, assign_labels = assign_labels,
            **spectral_args)


def data_cluster(cluster_model, train_data):
    """用指定的cluster_model聚类数据train_data
    [in]  cluster_model: object, 聚类模型
          train_data: array-like shape(n_samples, n_features), 待聚类数据
    [out] cluster_model: object, 聚类模型
    """
    logging.info("start MiniBatchKMeans")
    start_time = time.time()
    # 评估聚类效果依赖train_data和模型聚类结果
    cluster_model.fit(train_data)
    logging.info("cost time : %.4fs" % (time.time() - start_time))
    logging.info("score : %.4f" % cluster_score(train_data, cluster_model.labels_))
    return cluster_model
