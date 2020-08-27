#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model_impl.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/21 20:37:58
"""

import codecs
import logging
import numpy as np
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from sklearn.metrics import classification_report

from lr_model import LRModel
from preprocess import Preprocessor
from utils.data_io import write_to_file
from utils.for_def_user import LabelEncoder

class BaseLRModel(object):
    """LR����ģ�ͻ�����
    """
    def __init__(self, model_dir, output_dir):
        """
        """
        self.feature_id_path = os.path.join(model_dir, "feature_id.txt")
        self.class_id_path = os.path.join(model_dir, "class_id.txt")
        self.model_path = os.path.join(model_dir, "model.txt")
        self.feature_weight_path = os.path.join(model_dir, "feature_weight.txt")
        self.generator_path = os.path.join(model_dir, "generator.pkl")

        self.pred_res_path = os.path.join(output_dir, "pred_res.txt")
        self.wrong_pred_res_path = os.path.join(output_dir, "wrong_pred_res.txt")
        self.right_pred_res_path = os.path.join(output_dir, "right_pred_res.txt")

        self.label_encoder = LabelEncoder(self.class_id_path)
        self.label_thres = BaseLRModel.load_label_thres(self.class_id_path)
        self.lr_model = LRModel()
        logging.debug("\n".join(["%s:%d" % x for x in self.label_encoder.label_id_dict.items()]).encode("gb18030"))

    @staticmethod
    def load_label_thres(file_name):
        """
        [in]  file_name: str, ������id�ļ�, ����������, class_id, class_name, [thres]
        [out] class_thres: dict, str(class_id) -> float(thres)
        ���Ϊ���У���ʾ��class_id��Ҫ������ֵ
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
                # ���û�������Ŷ���ֵ ������
                continue
            pred_label = self.label_encoder.inverse_transform(int(pred_label))
            pred_res.append((pred_label, label_prob, label_evidence))

        if len(pred_res) == 0:
            pred_res.append((u"����", u"1.0", "NULL"))
        return pred_res

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

        _, train_data, train_label, val_data, val_label = preprocessor.gen_data_vec(
                data_dir,
                self.feature_id_path,
                split_train_test=split_train_test,
                feature_select=feature_select,
                to_file=to_file,
                re_seg=re_seg,
                process_file_path=mid_data_paths)

    def feature_label_gen(self, line):
        """�����ַ��� ��ȡ��������� ��ɶ�Ԫ��
        [in]  line: str, ���ݼ�ÿһ�е�����
        [out] res: (int, str), ��Ԫ��������������� �����ɿո�����Ϊ�ַ���
        """
        raise NotImplementedError("function feature_label_gen should be over written.")

    def train(self, train_lib_format_path, liblinear_train_path):
        """
        """
        self.lr_model.liblinear_train(
                train_lib_format_path,
                self.model_path,
                liblinear_train_path)

        self.lr_model.save_in_feature_weight_format(
                feature_weight_save_path=self.feature_weight_path,
                model_path=self.model_path,
                feature_id_path=self.feature_id_path)

    def eval(self, val_data_path):
        """
        """
        if not self.lr_model.model_loaded:
            logging.debug("model not loaded")
            self.lr_model.load_model(
                    model_path=self.model_path,
                    feature_id_path=self.feature_id_path)

        pred_label_list = list()
        real_label_list = list()
        pred_info_list = list()
        wrong_info_list = list()
        with codecs.open(val_data_path, "r", "gb18030") as rf:
            for line in rf:
                line = line.strip("\n")
                real_label, feature_str = self.feature_label_gen(line)
                features = feature_str.split(" ")
                pred_list = self.check(features)
                # get the first pred label
                pred_label_name, label_prob, label_evidence = pred_list[0]
                real_label_name = self.label_encoder.inverse_transform(int(real_label))
                real_label_list.append(real_label_name)
                pred_label_list.append(pred_label_name)
                info = "\t".join([pred_label_name, label_prob, line, label_evidence])
                pred_info_list.append(info)
                if pred_label_name != real_label_name:
                    wrong_info_list.append(info)
        write_to_file(pred_info_list, self.pred_res_path)
        write_to_file(wrong_info_list, self.wrong_pred_res_path)
        print(classification_report(real_label_list, pred_label_list, digits=4).encode("gb18030"))

        real_label_list = np.array(real_label_list)
        pred_label_list = np.array(pred_label_list)
        correct_num = sum(real_label_list==pred_label_list)
        total_num = len(real_label_list)
        acc = correct_num / float(total_num)
        print("acc = %.4f(%d/%d)" % (acc, correct_num, total_num))
        



if __name__ == "__main__":
    pass
