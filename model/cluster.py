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
    """��������Ч�� Խ��Խ��
    [in]  X: array-like, shape(n_samples, n_features), ���ݾ���
          y_pred: array-like, shape(n_samples,), ������
    [out] score: double, ����Ч���÷�
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
    """ ����kmeans �ɵ�����:n_clusters
    [in]  n_clusters: int, ����ظ���
          init: str, ��ʼ������
          batch_size: int, ��������ʱÿ����С
          n_init: int, �ظ��������
          max_no_improvement: int, ָ�����ٴ�û��������ֹͣ����
          verbose: int, Խ����չʾ����ϢԽ��
          **kmeans_args: dict, MiniBatchKMeans��������
    [out] MiniBatchKMeans, ����ģ��
    """
    return MiniBatchKMeans(n_clusters = n_clusters, init=init, batch_size=batch_size, n_init=n_init,
            max_no_improvement=max_no_improvement, verbose=verbose, **kmeans_args)


def spectral_clustering(n_clusters=None, gamma=None, assign_labels="kmeans", **spectral_args):
    """�׾��� �ɵ�����:n_clusters gamma
          ��ʹ��gridsearchcvʱ SpectralClusteringû��predict����
          ��Ϊ��̳е�ClusterMixin��û��predict���� ֻ��fit_predict����
          ��˽�ClusterMixin�е�fit_predict��Ϊ��predict
    [in]  n_clusters: int, ͶӰ�ӿռ��ά��
          gamma: double, �˺���rbf��kernel coefficient
          assign_labels: str, ��Ƕ��ռ�����ǩ�Ĳ���
          **spectral_args: dict, SpectralClustering������������
    [out] SpectralClustering, �׾���ģ��
    """
    return SpectralClustering(n_clusters = n_clusters, gamma = gamma, assign_labels = assign_labels,
            **spectral_args)


def data_cluster(cluster_model, train_data):
    """��ָ����cluster_model��������train_data
    [in]  cluster_model: object, ����ģ��
          train_data: array-like shape(n_samples, n_features), ����������
    [out] cluster_model: object, ����ģ��
    """
    logging.info("start MiniBatchKMeans")
    start_time = time.time()
    # ��������Ч������train_data��ģ�;�����
    cluster_model.fit(train_data)
    logging.info("cost time : %.4fs" % (time.time() - start_time))
    logging.info("score : %.4f" % cluster_score(train_data, cluster_model.labels_))
    return cluster_model
