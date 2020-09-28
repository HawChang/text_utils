#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lr_model.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/23 14:03:51
"""

import codecs
import logging
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from utils.data_io import read_from_file
from utils.data_io import dump_libsvm_file
from utils.softmax import softmax

from sklearn.feature_extraction.text import CountVectorizer

from tempfile import NamedTemporaryFile


def argsort(number_list):
    """排序由小到大，但返回的是下标排序结果
    例如：
        数组a=[3,5,4,2,1] 则argsort(a)=[4,3,0,2,1]：表示第4位是a数组中值最小的,第3位是a数组中第二小的
    [in] number_list: list[可直接比较大小的元素], 待排序列表
    """
    return sorted(range(len(number_list)), key=lambda x:number_list[x])

class LRModel(object):
    def __init__(self):
        """LR模型初始化
        """
        self.model_loaded = False

        #  LR模型所需
        self.label_list = None
        self.feature_name_list = None
        self.feature_weight_dict = None
        self.softmax_feature_weight_dict = None

    def train(self,
            train_feature,
            train_label,
            tmp_dir = "/tmp",
            token_pattern=r'(?u)[^ ]+',
            min_df = 2,
            liblinear_train_path="/home/users/zhanghao55/workspace/tools/liblinear-2.20/train",
            model_conf="-s 0",
            ):
        """训练LR模型
        [in]  train_feature: list[list[str]], 训练数据特征的列表
              train_label: list[int], 训练标签
              tmp_dir: str, 训练时临时文件存放的目录
              token_pattern: str, vectorizer用于划分token的模板
              min_df: int, 特征频率最小阈值
              liblinear_train_path: str, liblinear训练工具地址
              model_conf: str, 训练时参数
              例如:
              L2-regularized logistic regression (primal): -s 0 （默认训练参数）
              L1-regularized logistic regression: -s 6
              设定-s的同时可以加上类别权重调整class weight set: -w4 0.1
        """
        train_feature_str = [" ".join(features) for features in train_feature]
        # 根据train_feature生成liblinear格式的文件
        vectorizer = CountVectorizer(token_pattern=token_pattern, lowercase=False, min_df=min_df)
        # 生成训练特征
        feature_vec = vectorizer.fit_transform(train_feature_str)
        with NamedTemporaryFile(dir=tmp_dir) as libsvm_format_file:
            logging.info("libsvm format file: {}".format(libsvm_format_file.name))
            # 生成liblinear训练数据
            dump_libsvm_file(feature_vec, train_label, libsvm_format_file.name, False)
            with NamedTemporaryFile(dir=tmp_dir) as liblinear_model_file:
                logging.info("liblinear model file: {}".format(liblinear_model_file.name))
                # liblinear训练
                self.liblinear_train(
                        train_data_path=libsvm_format_file.name,
                        model_path=liblinear_model_file.name,
                        liblinear_train_path=liblinear_train_path,
                        model_conf=model_conf,
                        )
                # 生成预测所需权重dict
                self.gen_feature_weight_dict(liblinear_model_file.name, vectorizer.get_feature_names())

    def gen_feature_weight_dict(self, liblinear_model_path, feature_name_list):
        """根据liblinear生成的模型文件和特征保留文件生成线上需要的multiclass特征权重文件
        [in]  liblinear_model_path: str, liblinear模型文件地址
              feature_name_list: list(str), 特征字面列表, 其顺序应与liblinear模型文件中特征的顺序一致
        """

        self.label_list = list()
        self.feature_name_list = feature_name_list
        self.feature_weight_dict = dict()
        self.softmax_feature_weight_dict = dict()

        # 读入模型文件中的特征权值
        # 按类别值从小到大的顺序给出
        class_num = None
        feature_index = 0
        logging.info("gen feature weight file from liblinear model: %s." % liblinear_model_path)
        start_time = time.time()
        with codecs.open(liblinear_model_path, "r", "gb18030") as rf:
            for index, line in enumerate(rf):
                line = line.strip("\n")
                if index == 1:
                    class_num = int(line.split(" ")[-1])
                elif index == 2:
                    labels = [int(x) for x in line.split(" ")[1:]]
                    assert class_num == len(labels), "class num(%d) != labels size(%d)." % (class_num, len(labels))
                    # 数组a=[3,5,4,2,1] 则argsort(a)=[4,3,0,2,1]：表示第4位是a数组中值最小的,第3位是a数组中第二小的
                    # index_transfer是类别数组，各列按类别的值(类别由LabelEncoder编码后,为整数，可比较)排序的结果
                    # index_transfer[i]=j表示，第i小的类别是第j列
                    index_transfer = argsort(labels)
                    logging.info("position rank: " + ",".join([str(x) for x in index_transfer]))
                    for class_index in range(class_num):
                        self.label_list.append(str(labels[index_transfer[class_index]]))

                if index < 6:
                    continue

                # liblinear权值文件 最后会有一个空格
                weights = line.strip(" ").split(" ")
                assert len(weights) == class_num or class_num == 2, "wrong weight num at line #%d, expect %d, actual %d." \
                        % (index+1, class_num, len(weights))
                reordered_weights = list()
                if class_num > 2:
                    for weight_index in range(class_num):
                        reordered_weights.append(float(weights[index_transfer[weight_index]]))
                else:
                    # 二分类时 liblinear权重只有一维 该权重是第一个label的权重 该权重取反作为第二个label的权重
                    # 此时weights只有一列
                    reordered_weights = [0, 0]
                    # 权值是针对第一个label的
                    reordered_weights[index_transfer[0]] = float(weights[0])
                    # 第二个label为该权值取反
                    reordered_weights[index_transfer[1]] = -float(weights[0])

                self.feature_weight_dict[self.feature_name_list[feature_index]] = reordered_weights
                self.softmax_feature_weight_dict[self.feature_name_list[feature_index]] = softmax(reordered_weights, axis=1)
                feature_index += 1
        logging.info("cost time %.4fs." % (time.time() - start_time))
        self.model_loaded = True

    def save(self, model_path):
        """存储模型文件
        [in]  model_path: str, 模型存储文件
        """
        if not self.model_loaded:
            logging.error("model not loaded")

        logging.info("save model to: {}".format(model_path))
        start_time = time.time()
        with codecs.open(model_path, "w", "gb18030") as wf:
            wf.write("classes: %s" % ",".join(self.label_list))
            for index, feature_name in enumerate(self.feature_name_list):
                weight_str = " ".join(["%.20f" % x for x in self.feature_weight_dict[feature_name]])
                wf.write("\n" + "\t".join([str(index), feature_name, weight_str]))
        logging.info("cost time %.4fs." % (time.time() - start_time))

    def load(self, model_path):
        """加载模型文件
        [in]  model_path: str, 模型文件加载地址
        """
        logging.info("load model from: {}".format(model_path))
        start_time = time.time()

        self.label_list = None
        self.feature_name_list = list()
        self.feature_weight_dict = dict()
        self.softmax_feature_weight_dict = dict()

        with codecs.open(model_path, "r", "gb18030") as rf:
            for index, line in enumerate(rf):
                line = line.strip("\n")
                if index == 0:
                    self.label_list = line.split(" ")[-1].split(",")
                    class_num  = len(self.label_list)
                    assert class_num > 1, "class num should greater than 1, actual {}".format(class_num)
                    continue

                feature_id, feature_name, weights_str = line.strip("\n").split("\t")
                self.feature_name_list.append(feature_name)

                weights = [float(x) for x in weights_str.split(" ")]
                assert len(weights) == class_num, "wrong weight num at line #%d, expect %d, actual %d." \
                        % (index+1, class_num, len(weights))

                self.feature_weight_dict[feature_name] = weights
                self.softmax_feature_weight_dict[feature_name] = softmax(weights, axis=1)

        logging.info("cost time %.4fs." % (time.time() - start_time))

    def liblinear_train(self,
            train_data_path,
            model_path,
            liblinear_train_path="/home/users/zhanghao55/workspace/tools/liblinear-2.20/train",
            model_conf="-s 0"):
        """使用liblinear训练模型
        [in] train_data_path: str, 训练数据地址 liblinear训练数据格式
             model_path: str, 训练模型存储地址
             liblinear_train_path: str, liblinear训练工具地址
             model_conf: str, 训练时参数
             例如:
             L2-regularized logistic regression (primal): -s 0 （默认训练参数）
             L1-regularized logistic regression: -s 6
             设定-s的同时可以加上类别权重调整class weight set: -w4 0.1
        """
        # liblinear工具地址
        assert os.path.isfile(liblinear_train_path), "%s is not a file." % liblinear_train_path
        try:
            logging.info("liblinear train start...")
            start_time = time.time()
            cmd_str =' '.join([
                    liblinear_train_path,
                    model_conf,
                    train_data_path,
                    model_path])
            logging.debug("cmd str: %s" % cmd_str)

            with subprocess.Popen(
                    cmd_str,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1) as popen:

                # 重定向标准输出
                while popen.poll() is None:
                    # None表示正在执行中
                    r = popen.stdout.readline().decode().strip("\n")
                    if len(r) > 0:
                        logging.debug(r)

                # 重定向错误输出
                if popen.poll() != 0:
                    # 不为0表示执行错误
                    err = popen.stderr.readline().decode().strip("\n")
                    if len(err) > 0:
                        logging.error(err)

            logging.info("cost time %.4fs." % (time.time() - start_time))

        except subprocess.CalledProcessError as e:
            logging.warning("liblinear train failed.")
            logging.error(e)
            raise e

        logging.info("liblinear train finish")

    def predict(self, features_list, **kwargs):
        """预测多个文本
        [in]  features_list : list[list[str]], 多个文本的特征列表
              min_conf : float, 作为标签的最小置信度
              digits: int, 精度控制在几位小数
              evidence: bool, true则多返回一个正向特征信息 固定返回
              topk: int, 取绝对值前topk个特征作为证据
        [out] pred_list: list[list[(str, str, str)]], 多个预测结果
                         每个结果为二元组列表, (类别, 置信度, 证据) 由大到小
        **已验证 与liblinear模型输出概率一致 此时bias为-1（及默认值）
        """
        pred_res = list()
        for features in features_list:
            cur_res = self._single_predict(features, **kwargs)
            pred_res.append(cur_res)
        return pred_res

    def _single_predict(self, features, min_conf=0.05, digits=4, evidence=False, topk=10):
        """根据特征列表预测结果
        [in]  features : list[str], 特征列表
              min_conf : float, 作为标签的最小置信度
              digits: int, 精度控制在几位小数
              evidence: bool, true则多返回一个正向特征信息 固定返回
              topk: int, 取绝对值前topk个特征作为证据
        [out] pred_list: list[(str, str)], 预测结果二元组列表, (类别, 置信度) 由大到小
        **已验证 与liblinear模型输出概率一致 此时bias为-1（及默认值）
        """
        label_value = defaultdict(lambda: 0.0)
        total = 0.0
        evidence_dict = defaultdict(list)
        #hit_feature = set(features) & set(self.feature_weight_dict.keys())
        #print("hit features: {}".format(" ".join(sorted(hit_feature)).encode("gb18030")))
        for feature in features:
            if feature not in self.feature_weight_dict:
                continue
            softmax_weights = self.softmax_feature_weight_dict[feature]
            for index, weight in enumerate(self.feature_weight_dict[feature]):
                try:
                    label_value[self.label_list[index]] += weight
                    # 用softmax的结果来判断各特征对当前类的重要性 但给证据时要给实际该特征对该类的值
                    evidence_dict[self.label_list[index]].append((feature, softmax_weights[index], weight))
                except IndexError as e:
                    print("index error. index = %d." % index)
                    raise e
    
        #logging.debug("label_weght_sum: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value) for label,value in label_value.items() ]))
    
        for label, sum_weight in label_value.items():
            label_value[label] = 1.0 / (1.0 + math.exp(-sum_weight))
            total += label_value[label]
    
        #logging.debug("label_value: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value) for label,value in label_value.items() ]))
        digits_format = "%%.%df" % digits
        pred_list = list()
        for label, weight_pred in sorted(label_value.items(), key=lambda x: x[1], reverse=True):
            pred_proba = weight_pred / total
            if pred_proba < min_conf:
                continue
            # 准备证据
            cur_label_evidence = "||".join(["%s(%.6f)" % (x[0], x[2]) for x in \
                    sorted(evidence_dict[label], key=lambda x:abs(x[1]), reverse=True)[:topk]])
            pred_list.append((label, digits_format % pred_proba, cur_label_evidence))
        
        #label_evidence = None
        #if evidence:
        #    label_evidence = u"无预测结果" if len(pred_list) == 0 else \
        #            "||".join(["%s(%.6f)" % (x[0], x[2]) for x in \
        #            sorted(evidence_dict[pred_list[0][0]], key=lambda x:x[1], reverse=True)[:topk]])
        return pred_list


if __name__ == "__main__":
    pass
