#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model_multilabel_impl.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2020/06/03 20:05:39
"""

import codecs
import json
import logging
import os
import sys
import time
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from sklearn.metrics import classification_report

from lr_model_multilabel import LRModelMultiLabel
from preprocess import Preprocessor
from utils.data_io import write_to_file
from utils.data_io import dump_pkl
from utils.data_io import load_pkl
from utils.for_def_user import LabelEncoder
from utils.multi_label_metrics import multilabel_classification_report, multilabel_classification_diff

class BaseLRModel(object):
    """LR分类模型基础类
    """
    def __init__(self, model_dir, output_dir):
        """
        """
        self.feature_id_path = os.path.join(model_dir, "feature_id.txt")
        self.class_id_path = os.path.join(model_dir, "class_id.txt")
        self.model_path = os.path.join(model_dir, "model")
        self.feature_weight_path = os.path.join(model_dir, "feature_weight.txt")
        self.vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        self.pred_res_path = os.path.join(output_dir, "pred_res.txt")
        self.wrong_pred_res_path = os.path.join(output_dir, "wrong_pred_res.txt")
        self.right_pred_res_path = os.path.join(output_dir, "right_pred_res.txt")

        self.label_encoder = LabelEncoder(self.class_id_path)
        self.label_thres = BaseLRModel.load_label_thres(self.class_id_path)
        self.lr_model_mult = LRModelMultiLabel()
        logging.debug("\n".join(["%s:%d" % x for x in self.label_encoder.label_id_dict.items()]).encode("gb18030"))

    @staticmethod
    def load_label_thres(file_name):
        """
        [in]  file_name: str, 分类名id文件, 两列至三列, class_id, class_name, [thres]
        [out] class_thres: dict, str(class_id) -> float(thres)
        如果为三列，表示该class_id需要限制阈值
        """
        class_thres = dict()
        with codecs.open(file_name, "r", "gb18030") as rf:
            for eachline in rf:
                parts = eachline.strip().split("\t")
                if len(parts) == 3:
                    class_thres[parts[0]] = float(parts[2])
        return class_thres

    def check(self, features, evidence=True):
        pred_list = self.lr_model.check(features, evidence=evidence)

        pred_res = list()
        for pred_label, label_prob, label_evidence in pred_list:
            if float(label_prob) < self.label_thres.get(pred_label, 0.5):
                # 如果没超过置信度阈值 则跳过
                continue
            pred_label = self.label_encoder.inverse_transform(int(pred_label))
            pred_res.append((pred_label, label_prob, label_evidence))

        if len(pred_res) == 0:
            pred_res.append((u"其他", u"1.0", "NULL"))
        return pred_res

    def preprocess(self,
            data_dir,
            re_seg=True,
            to_file=False,
            libsvm_format=False,
            mid_data_paths=None,
            split_train_test=True,
            test_ratio=0.2,
            vec_method="count",
            feature_select=False,
            is_percent=True,
            feature_keep_percent=90,
            feature_keep_num=10,
            min_df=3):
        """
        """
        preprocessor = Preprocessor(
                feature_gen_func=self.feature_label_gen,
                vec_method=vec_method,
                feature_keep_percent=feature_keep_percent,
                feature_keep_num=feature_keep_num,
                is_percent=is_percent, 
                test_ratio=test_ratio,
                min_df=min_df)

        self.vectorizer, train_data, train_label, val_data, val_label = preprocessor.gen_data_vec(
                data_dir,
                self.feature_id_path,
                split_train_test=split_train_test,
                feature_select=feature_select,
                to_file=to_file,
                libsvm_format=libsvm_format,
                re_seg=re_seg,
                process_file_path=mid_data_paths)

        # 保存vectorizer
        dump_pkl(self.vectorizer, self.vectorizer_path, True)

    def feature_label_gen(self, line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        raise NotImplementedError("function feature_label_gen should be over written.")

    def train(self, train_pkl_path):
        """
        """
        logging.info("load train data")
        start_time = time.time()
        # train_label此时为json字符串
        train_feature_vec, train_label = load_pkl(train_pkl_path)
        # 将每个train_label转为list
        train_label = [json.loads(x) for x in train_label]
        logging.info("cost_time : %.4f" % (time.time() - start_time))

        self.lr_model_mult.train(
                train_feature_vec,
                train_label)

        self.lr_model_mult.save(self.model_path, overwrite=True)

    def eval(self, val_data_path):
        """
        """
        if not self.lr_model_mult.model_loaded:
            logging.debug("model not loaded")
            self.lr_model_mult.load_model(self.model_path)

        if not hasattr(self, "vectorizer"):
            self.vectorizer = load_pkl(self.vectorizer_path)

        pred_label_list = list()
        real_label_list = list()
        pred_info_list = list()
        total_pred_time = 0
        with codecs.open(val_data_path, "r", "gb18030") as rf:
            for line in rf:
                line = line.strip("\n")
                real_label, feature_str = self.feature_label_gen(line)
                real_label = json.loads(real_label)
                real_label_list.append(real_label)
                feature_vec = self.vectorizer.transform([feature_str])
                start_time = time.time()
                pred_label = list(self.lr_model_mult.check(feature_vec)[0])
                total_pred_time += time.time() - start_time
                pred_label_list.append(pred_label)
                pred_info_list.append("\t".join([json.dumps(pred_label), line]))
        write_to_file(pred_info_list, self.pred_res_path)
        print("predict num = {}, speed = {:.4f}s/example"\
                .format(len(pred_info_list), total_pred_time/float(len(pred_info_list))))

        multilabel_classification_report(pred_label_list, real_label_list, self.label_encoder)


if __name__ == "__main__":
    pass
