#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: cluster_model_impl.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2020/03/23 11:25:43
"""

import logging
import os
import sys
import time
import warnings
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from sklearn.metrics import make_scorer

from cluster import cluster_score
from cluster import data_cluster
from cluster import mini_batch_kmeans
from utils.data_io import get_data
from utils.data_io import read_from_file
from utils.data_io import write_to_file
from utils.parameter_optimizer import grid_search_cv
from feature.feature_generator import FeatureGenerator
from preprocess import Preprocessor

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

duplicate = False
re_seg = True

class BaseCluster(object):
    def __init__(self, model_dir, output_dir):
        """
        """
        self.feature_id_path = os.path.join(model_dir, "feature_id.txt")
        self.cluster_model_path = os.path.join(model_dir, 'cluster_model.pkl')
        self.generator_path = os.path.join(model_dir, "generator.pkl")

        self.cluster_res_path = os.path.join(output_dir, "cluster_res.txt")

        self.line_process_num = 0
        logging.info("BaseCluster init succeed")

    def feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        raise NotImplementedError("function feature_label_gen should be over written.")

    def preprocess(self,
            data_dir,
            re_seg=True,
            to_file=False,
            mid_data_paths=None,
            split_train_test=True,
            test_ratio=0.2,
            vec_method="count",
            feature_select=True,
            is_percent=True,
            feature_keep_percent=90,
            feature_keep_num=10,
            min_df=3):
        """根据指定目录 获得数据特征
        [out] train_data_vec: matrix, 数据集特征
        """
        preprocessor = Preprocessor(
                feature_gen_func=self.feature_label_gen,
                vec_method=vec_method,
                feature_keep_percent=feature_keep_percent,
                feature_keep_num=feature_keep_num,
                is_percent=is_percent, 
                test_ratio=test_ratio,
                min_df=min_df)

        # 根据数据生成特征
        _, self.train_data_vec, _, _, _ = preprocessor.gen_data_vec(
                data_dir,
                self.feature_id_path,
                split_train_test=split_train_test,
                feature_select=feature_select,
                to_file=to_file,
                re_seg=re_seg,
                process_file_path=mid_data_paths)


    def cluster(self,
            data_path,
            n_clusters=100,
            params={'n_clusters': [5, 10, 20, 50, 75, 100]},
            grid_search=True):
        """根据数据特征进行聚类
        """
        self.preprocess(data_path)
        if grid_search:
            cluster_model = mini_batch_kmeans(n_clusters=None, max_no_improvement=10000)
            gsc = grid_search_cv(
                    estimator = cluster_model,
                    #params = {"n_clusters": [6,8,10,12,14,16]},
                    params = params,
                    train_data = self.train_data_vec,
                    test_data = self.train_data_vec,
                    scoring = make_scorer(cluster_score, greater_is_better=True),
                    n_jobs = 15,
                    refit = True,
                    cv = 5)
            cluster_model = gsc.best_estimator_
        else:
            cluster_model = mini_batch_kmeans(n_clusters=n_clusters, max_no_improvement=10000)
            data_cluster(cluster_model, self.train_data_vec)

        logging.debug("score : %.4f" % cluster_score(self.train_data_vec, cluster_model.labels_))
        cluster_ids = cluster_model.labels_
        train_data = get_data(data_path)
        if self.cluster_res_path is not None:
            write_to_file(sorted(zip(cluster_ids, train_data), key=lambda x:x[0]), self.cluster_res_path, write_func=lambda x: "%s\t%s" % x)


if __name__ == "__main__":
    pass

