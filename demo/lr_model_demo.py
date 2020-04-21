#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model_demo.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/21 20:37:58
"""

import logging
import os
import re
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from model.lr_model_impl import BaseLRModel
from preprocess import ProcessFilePath
from feature.feature_generator import FeatureGenerator
from utils.logger import init_log
init_log("./log/lr_model.log")

import config


class LRModelDemo(BaseLRModel):
    """LR分类模型基础类
    """
    def __init__(self, mid_data_dir, model_dir, output_dir):
        """
        """
        super(LRModelDemo, self).__init__(model_dir, output_dir)
        self.mid_data_paths = ProcessFilePath(output_dir=mid_data_dir)

        self.feature_generator = FeatureGenerator(
                seg_method=config.seg_method,
                segdict_path=config.segdict_path,
                stopword_path=config.stopword_path,
                ngram=config.ngram,
                feature_min_length=config.feature_min_length)
        FeatureGenerator.save(self.feature_generator, self.generator_path, True)
        self.line_process_num = 0

    def train_feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        parts = line.strip("\n").split("\t")
        label = self.label_encoder.transform(parts[0].strip())
        idea_list = parts[1].split("\x01")
        word_list = parts[2].split("\x01")
        feature_list = list()
        if config.idea_word_feature_sep:
            for text in idea_list:
                feature_list.extend(["idea_%s" % x for x in self.feature_generator.gen_feature(text, duplicate=config.duplicate)])
            for text in word_list:
                feature_list.extend(["word_%s" % x for x in self.feature_generator.gen_feature(text, duplicate=config.duplicate)])
        else:
            for text in idea_list + word_list:
                feature_list.extend(self.feature_generator.gen_feature(text, duplicate=config.duplicate))

        if self.line_process_num % 4000 == 0:
            text = "||".join(parts[1:3])
            text = re.sub(config.rex, "", text).strip()
            seg_text = "/ ".join(self.feature_generator.seg_words(text))
            log.debug("process line num #%d" % self.line_process_num)
            log.debug("origin  : %s" % text.encode("gb18030"))
            log.debug("="*150)
            log.debug("seg res : %s" % seg_text.encode("gb18030"))
        features = feature_list if config.duplicate else set(feature_list)
        self.line_process_num += 1
        return (label, " ".join(features))

    def preprocess(self, data_dir):
        """
        """
        self.feature_label_gen = self.train_feature_label_gen
        self.line_process_num = 0
        super(LRModelDemo, self).preprocess(
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

    def train(self):
        """
        """
        super(LRModelDemo, self).train(
                self.mid_data_paths.train_lib_format_path,
                config.liblinear_train_path)

    def eval(self):
        """
        """
        self.feature_label_gen = self.train_feature_label_gen
        self.line_process_num = 0
        super(LRModelDemo, self).eval(self.mid_data_paths.val_data_path)


if __name__ == "__main__":
    labeler = LRModelDemo(
            mid_data_dir=config.mid_data_dir,
            model_dir=config.model_dir,
            output_dir=config.output_dir)
    labeler.preprocess(
            data_dir=config.train_data_dir)
    labeler.train()
    labeler.eval()
