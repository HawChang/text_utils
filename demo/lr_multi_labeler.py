#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_multi_labeler.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/21 20:37:58
"""

import codecs
import os
import sys
import time
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from model.lr_model import LRModel
from preprocess import Preprocessor
from utils.data_io import read_from_file
from utils.data_io import write_to_file
from feature.feature_generator import FeatureGenerator
from utils.file_manager import FileManager
from utils.for_def_user import LabelEncoder
from utils.logger import Logger

log = Logger().get_logger()

import config

class BaseLRLabeler(object):
    """LR分类模型基础类
    """
    def __init__(self):
        """初始化
        """
        self.f_manager = FileManager(
                data_root=config.data_root,
                model_root=config.model_root)
        
        self.label_encoder = LabelEncoder(self.f_manager.class_id_path)
        self.lr_model = LRModel(self.f_manager)
        #print("\n".join(["%s:%d" % x for x in self.label_encoder.label_id_dict.items()]).encode("gb18030"))
        
        self.feature_generator = FeatureGenerator(
                seg_method=config.seg_method,
                stopword_path=config.stopword_path,
                ngram=config.ngram,
                feature_min_length=config.feature_min_length)
        FeatureGenerator.save(self.feature_generator, self.f_manager.generator_path, True)
        self.line_process_num = 0
        self.model_loaded = False

    def preprocess(self):

        preprocessor = Preprocessor(
                file_manager=self.f_manager,
                feature_gen_func=self.feature_label_gen,
                vec_method=config.vec_method,
                feature_keep_percent=config.feature_keep_percent,
                feature_keep_num=config.feature_keep_num,
                is_percent=config.is_percent, 
                test_ratio=config.test_ratio,
                min_df=config.min_df,
                re_seg=config.re_seg)

        # 根据数据生成特征
        train_data, train_label, val_data, val_label = \
                preprocessor.gen_data_vec(config.train_data_root, split_train_test=config.split_train_test, feature_select=config.feature_select)

    def feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        parts = line.strip("\n").split("\t")
        label = self.label_encoder.transform(parts[0].strip())
        idea_list = parts[1].split("\x01")
        word_list = parts[2].split("\x01")
        feature_list = list()
        self.line_process_num += 1
        if config.idea_word_feature_sep:
            for text in idea_list:
                feature_list.extend(["idea_%s" % x for x in self.feature_generator.gen_feature(text, duplicate=config.duplicate)])
            for text in word_list:
                feature_list.extend(["word_%s" % x for x in self.feature_generator.gen_feature(text, duplicate=config.duplicate)])
        else:
            for text in idea_list + word_list:
                feature_list.extend(self.feature_generator.gen_feature(text, duplicate=onfig.duplicate))

        if self.line_process_num % 4000 == 0:
            text = "||".join(parts[1:3])
            seg_text = "/ ".join(self.feature_generator.seg_words(text))
            log.debug("process line num #%d" % self.line_process_num)
            log.debug("origin  : %s" % text.encode("gb18030"))
            log.debug("="*150)
            log.debug("seg res : %s" % seg_text.encode("gb18030"))
        features = feature_list if config.duplicate else set(feature_list)
        return (label, " ".join(features))

    def train(self):
        """
        """
        self.lr_model.liblinear_train()
        self.lr_model.save_in_feature_weight_format()

    def load(self):
        self.lr_model.load_model()
        self.model_loaded = True

    def eval(self):
        """
        """
        if not self.model_loaded:
            self.load()
        pred_label_list = list()
        real_label_list = list()
        pred_info_list = list()
        wrong_info_list = list()
        with codecs.open(self.f_manager.val_data_path, "r", "gb18030") as rf:
            for line in rf:
                real_label, feature_str = self.feature_label_gen(line)
                features = feature_str.split(" ")
                label_res = self.lr_model.check(features)
                if len(label_res) == 0:
                    pred_label = u"其他"
                    label_prob = u"1.0"
                else:
                    pred_label, label_prob = label_res[0]
                    pred_label = self.label_encoder.inverse_transform(int(pred_label))
                real_label = self.label_encoder.inverse_transform(int(real_label))
                real_label_list.append(real_label)
                pred_label_list.append(pred_label)
                info = "\t".join([pred_label, label_prob, data])
                pred_info_list.append(info)
                if pred_label != real_label:
                    wrong_info_list.append(info)
        write_to_file(pred_info_list, self.f_manager.pred_res_path)
        write_to_file(wrong_info_list, self.f_manager.wrong_pred_res_path)
        print(classification_report(real_label_list, pred_label_list, digits=4).encode("gb18030"))


if __name__ == "__main__":
    labeler = BaseLRLabeler()
    #labeler.preprocess()
    #labeler.train()
    labeler.eval()
