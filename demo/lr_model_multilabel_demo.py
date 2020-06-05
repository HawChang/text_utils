#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model_multilabel_demo.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2020/06/04 15:59:43
"""

import logging
import json
import os
import re
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from text_utils.model.lr_model_multilabel_impl import BaseLRModel
from text_utils.preprocess import ProcessFilePath
from text_utils.feature.feature_generator import FeatureGenerator
from text_utils.utils.logger import init_log
init_log("./log/lr_model_multilabel.log")

import lr_multiple_config as config


class LRModelMultipleDemo(BaseLRModel):
    """LR多标签分类模型基础类
    """
    def __init__(self, mid_data_dir, model_dir, output_dir):
        """
        """
        super(LRModelMultipleDemo, self).__init__(model_dir, output_dir)
        self.mid_data_paths = ProcessFilePath(output_dir=mid_data_dir)
        self.generator_path = os.path.join(model_dir, "generator.pkl")

        self.feature_generator = FeatureGenerator(
                seg_method=config.seg_method,
                segdict_path=config.segdict_path,
                stopword_path=config.stopword_path,
                ngram=config.ngram,
                feature_min_length=config.feature_min_length)

    def train_feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        parts = line.strip("\n").split("\t")
        label = parts[0]
        text = parts[1]
        text = re.sub(config.rex, "", text).strip()
        feature_list = self.feature_generator.gen_feature(text, duplicate=config.duplicate)

        if self.line_process_num % 4000 == 0:
            seg_text = "/ ".join(feature_list)
            logging.debug("process line num #%d" % self.line_process_num)
            logging.debug("origin  : %s" % text.encode("gb18030"))
            logging.debug("="*150)
            logging.debug("seg res : %s" % seg_text.encode("gb18030"))
        features = feature_list if config.duplicate else set(feature_list)
        self.line_process_num += 1
        return (label, " ".join(features))

    def preprocess(self, data_dir):
        """
        """
        self.feature_label_gen = self.train_feature_label_gen
        # 预处理时保存特征生成类
        FeatureGenerator.save(self.feature_generator, self.generator_path, True)
        # 初始化处理行数
        self.line_process_num = 0
        super(LRModelMultipleDemo, self).preprocess(
                data_dir,
                re_seg=config.re_seg,
                to_file=config.to_file,
                libsvm_format=config.libsvm_format,
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

    def train(self):
        """
        """
        super(LRModelMultipleDemo, self).train(self.mid_data_paths.train_pkl_path)

    def eval(self):
        """
        """
        self.feature_label_gen = self.train_feature_label_gen
        self.line_process_num = 0
        super(LRModelMultipleDemo, self).eval(self.mid_data_paths.val_data_path)


if __name__ == "__main__":
    labeler = LRModelMultipleDemo(
            mid_data_dir=config.mid_data_dir,
            model_dir=config.model_dir,
            output_dir=config.output_dir)
    labeler.preprocess(
            data_dir=config.train_data_dir)
    labeler.train()
    labeler.eval()
