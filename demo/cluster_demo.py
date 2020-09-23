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
Date: 2019/12/10 15:26:16
"""

import logging
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from text_utils.model.cluster_model_impl import BaseCluster
from text_utils.feature.feature_generator import FeatureGenerator
from text_utils.preprocess import ProcessFilePath
from text_utils.utils.logger import init_log
init_log("./log/cluster.log")

import config


class ClusterDemo(BaseCluster):
    def __init__(self, mid_data_dir, model_dir, output_dir):
        super(ClusterDemo, self).__init__(model_dir, output_dir)
        self.mid_data_paths = ProcessFilePath(output_dir=mid_data_dir)

        self.feature_generator = FeatureGenerator(
                seg_method=config.seg_method,
                segdict_path=config.segdict_path,
                stopword_path=config.stopword_path,
                ngram=config.ngram,
                feature_min_length=config.feature_min_length)

        FeatureGenerator.save(self.feature_generator, self.generator_path, True)
        self.line_process_num = 0
        logging.info("ClusterDemo init succeed")

    def cluster_feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        parts = line.strip("\n").split("\t")
        text = parts[7] + parts[8]
        self.line_process_num += 1
        feature_list = self.feature_generator.gen_feature(text, duplicate=config.duplicate)
        if self.line_process_num % 4000 == 0:
            seg_text = "/ ".join(self.feature_generator.seg_words(text))
            logging.debug("process line num #%d" % self.line_process_num)
            logging.debug("origin  : %s" % text.encode("gb18030"))
            logging.debug("="*150)
            logging.debug("seg res : %s" % seg_text.encode("gb18030"))
        features = feature_list if config.duplicate else set(feature_list)
        return (0, " ".join(features))

    def preprocess(self, data_dir):
        """根据指定目录 获得数据特征
        [out] train_data_vec: matrix, 数据集特征
        """
        self.feature_label_gen = self.cluster_feature_label_gen
        self.line_process_num = 0

        super(ClusterDemo, self).preprocess(
                data_dir,
                re_seg=config.re_seg,
                to_file=config.to_file,
                mid_data_paths=self.mid_data_paths,
                split_train_test=config.split_train_test,
                test_ratio=config.test_ratio,
                vec_method=config.vec_method,
                feature_select=config.feature_select,
                is_percent=config.is_percent,
                feature_keep_percent=config.feature_keep_percent,
                feature_keep_num=config.feature_keep_num,
                min_df=config.min_df,
                )

    def cluster(self,
            data_path,
            n_clusters=100,
            params={'n_clusters': [5, 10, 20, 50, 75, 100]},
            grid_search=True):
        """根据数据特征进行聚类
        """
        super(ClusterDemo, self).cluster(data_path, n_clusters, params, grid_search)


def main():
    cluster = ClusterDemo(
            mid_data_dir=config.mid_data_dir,
            model_dir=config.model_dir,
            output_dir=config.output_dir)
    cluster.cluster(data_path=config.train_data_dir, grid_search=False)


if __name__ == "__main__":
    main()

