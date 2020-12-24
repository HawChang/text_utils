#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: logistic_regression.py
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from tempfile import NamedTemporaryFile

_cur_dir = os.path.dirname(os.path.abspath(__file__))
from text_utils.utils.data_io import dump_libsvm_file
from text_utils.utils.softmax import softmax


def arg_sort(number_list):
    """������С���󣬵����ص����±�������
    ���磺
        ����a=[3,5,4,2,1] ��arg_sort(a)=[4,3,0,2,1]����ʾ��4λ��a������ֵ��С��,��3λ��a�����еڶ�С��
    [in] number_list: list[��ֱ�ӱȽϴ�С��Ԫ��], �������б�
    """
    return sorted(range(len(number_list)), key=lambda x:number_list[x])


class LogisticRegression(object):
    def __init__(self):
        """LRģ�ͳ�ʼ��
        """
        self.model_loaded = False

        #  LRģ������
        self.label_list = None
        self.feature_name_list = None
        self.feature_weight_dict = None
        self.softmax_feature_weight_dict = None

    def train(self, train_feature, train_label,
              liblinear_train_path=None,
              tmp_dir="/tmp", token_pattern=r'(?u)[^ ]+', min_df=2,
              model_conf="-s 0"):
        """ѵ��LRģ��
        [in]  train_feature: list[list[str]], ѵ�������������б�
              train_label: list[int], ѵ����ǩ
              liblinear_train_path: str, liblinearѵ�����ߵ�ַ, None��ʹ��Ĭ�ϵ�ַ
              tmp_dir: str, ѵ��ʱ��ʱ�ļ���ŵ�Ŀ¼
              token_pattern: str, vectorizer���ڻ���token��ģ��
              min_df: int, ����Ƶ����С��ֵ
              model_conf: str, ѵ��ʱ����
              ����:
              L2-regularized logistic regression (primal): -s 0 ��Ĭ��ѵ��������
              L1-regularized logistic regression: -s 6
              �趨-s��ͬʱ���Լ������Ȩ�ص���class weight set: -w4 0.1
        """
        if liblinear_train_path is None:
            liblinear_train_path = os.path.join(_cur_dir, "liblinear/train")
        train_feature_str = [" ".join(features) for features in train_feature]
        # ����train_feature����liblinear��ʽ���ļ�
        vectorizer = CountVectorizer(token_pattern=token_pattern, lowercase=False, min_df=min_df)
        # ����ѵ������
        feature_vec = vectorizer.fit_transform(train_feature_str)
        with NamedTemporaryFile(dir=tmp_dir) as libsvm_format_file:
            logging.info("libsvm format file: {}".format(libsvm_format_file.name))
            # ����liblinearѵ������
            dump_libsvm_file(feature_vec, train_label, libsvm_format_file.name, False)
            with NamedTemporaryFile(dir=tmp_dir) as liblinear_model_file:
                logging.info("liblinear model file: {}".format(liblinear_model_file.name))
                # liblinearѵ��
                self.liblinear_train(
                        train_data_path=libsvm_format_file.name,
                        model_path=liblinear_model_file.name,
                        liblinear_train_path=liblinear_train_path,
                        model_conf=model_conf,
                        )
                # ����Ԥ������Ȩ��dict
                self.gen_feature_weight_dict(liblinear_model_file.name, vectorizer.get_feature_names())

    def gen_feature_weight_dict(self, liblinear_model_path, feature_name_list):
        """����liblinear���ɵ�ģ���ļ������������ļ�����������Ҫ��multiclass����Ȩ���ļ�
        [in]  liblinear_model_path: str, liblinearģ���ļ���ַ
              feature_name_list: list(str), ���������б�, ��˳��Ӧ��liblinearģ���ļ���������˳��һ��
        """

        self.label_list = list()
        self.feature_name_list = feature_name_list
        self.feature_weight_dict = dict()
        self.softmax_feature_weight_dict = dict()

        # ����ģ���ļ��е�����Ȩֵ
        # �����ֵ��С�����˳�����
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
                    # ����a=[3,5,4,2,1] ��arg_sort(a)=[4,3,0,2,1]����ʾ��4λ��a������ֵ��С��,��3λ��a�����еڶ�С��
                    # index_transfer��������飬���а�����ֵ(�����LabelEncoder�����,Ϊ�������ɱȽ�)����Ľ��
                    # index_transfer[i]=j��ʾ����iС������ǵ�j��
                    index_transfer = arg_sort(labels)
                    logging.info("position rank: " + ",".join([str(x) for x in index_transfer]))
                    for class_index in range(class_num):
                        self.label_list.append(str(labels[index_transfer[class_index]]))

                if index < 6:
                    continue

                # liblinearȨֵ�ļ� ������һ���ո�
                weights = line.strip(" ").split(" ")
                assert len(weights) == class_num or class_num == 2, \
                    "wrong weight num at line #%d, expect %d, actual %d." % (index+1, class_num, len(weights))
                reordered_weights = list()
                if class_num > 2:
                    for weight_index in range(class_num):
                        reordered_weights.append(float(weights[index_transfer[weight_index]]))
                else:
                    # ������ʱ liblinearȨ��ֻ��һά ��Ȩ���ǵ�һ��label��Ȩ�� ��Ȩ��ȡ����Ϊ�ڶ���label��Ȩ��
                    # ��ʱweightsֻ��һ��
                    reordered_weights = [0, 0]
                    # Ȩֵ����Ե�һ��label��
                    reordered_weights[index_transfer[0]] = float(weights[0])
                    # �ڶ���labelΪ��Ȩֵȡ��
                    reordered_weights[index_transfer[1]] = -float(weights[0])

                self.feature_weight_dict[self.feature_name_list[feature_index]] = reordered_weights
                self.softmax_feature_weight_dict[self.feature_name_list[feature_index]] = \
                    softmax(reordered_weights, axis=1)
                feature_index += 1
        logging.info("cost time %.4fs." % (time.time() - start_time))
        self.model_loaded = True

    def save(self, model_path):
        """�洢ģ���ļ�
        [in]  model_path: str, ģ�ʹ洢�ļ�
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
        """����ģ���ļ�
        [in]  model_path: str, ģ���ļ����ص�ַ
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
                    self.label_list = [int(x) for x in line.split(" ")[-1].split(",")]
                    class_num = len(self.label_list)
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

    def liblinear_train(self, train_data_path, model_path,
                        liblinear_train_path="./liblinear/train", model_conf="-s 0"):
        """ʹ��liblinearѵ��ģ��
        [in] train_data_path: str, ѵ�����ݵ�ַ liblinearѵ�����ݸ�ʽ
             model_path: str, ѵ��ģ�ʹ洢��ַ
             liblinear_train_path: str, liblinearѵ�����ߵ�ַ
             model_conf: str, ѵ��ʱ����
             ����:
             L2-regularized logistic regression (primal): -s 0 ��Ĭ��ѵ��������
             L1-regularized logistic regression: -s 6
             �趨-s��ͬʱ���Լ������Ȩ�ص���class weight set: -w4 0.1
        """
        # liblinear���ߵ�ַ
        assert os.path.isfile(liblinear_train_path), "%s is not a file." % liblinear_train_path
        try:
            logging.info("liblinear train start...")
            start_time = time.time()
            cmd_str = ' '.join([
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

                # �ض����׼���
                while popen.poll() is None:
                    # None��ʾ����ִ����
                    r = popen.stdout.readline().decode().strip("\n")
                    if len(r) > 0:
                        logging.debug(r)

                # �ض���������
                if popen.poll() != 0:
                    # ��Ϊ0��ʾִ�д���
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
        """Ԥ�����ı�
        [in]  features_list : list[list[str]], ����ı��������б�
              min_conf : float, ��Ϊ��ǩ����С���Ŷ�
              digits: int, ���ȿ����ڼ�λС��
              evidence: bool, true��෵��һ������������Ϣ �̶�����
              topk: int, ȡ����ֵǰtopk��������Ϊ֤��
        [out] pred_list: list[list[(str, str, str)]], ���Ԥ����
                         ÿ�����Ϊ��Ԫ���б�, (���, ���Ŷ�, ֤��) �ɴ�С
        **����֤ ��liblinearģ���������һ�� ��ʱbiasΪ-1����Ĭ��ֵ��
        """
        pred_res = list()
        for features in features_list:
            cur_res = self._single_predict(features, **kwargs)
            pred_res.append(cur_res)
        return pred_res

    def eval(self, eval_feature, eval_label, eval_text_info=None, default_label=0,
             label_trans_func=None, eval_res_path=None, eval_diff_path=None):
        """����eval���ݼ�����ģ��Ч��
        [in]  eval_data: list[list[str]], �����б����б�
              eval_label: list[Any], ��ǩ�б�
              eval_text_info: list[str], eval�����ı�
              label_trans_func: function, ת��ģ�������ǩ�ĺ�����None��ת��
              eval_res_path: str, eval�����ַ
              eval_diff_path: str, eval�����diff��¼�ĵ�ַ
        """
        pred_res = self.predict(eval_feature)
        wf_list = list()
        try:
            eval_res_wf = None
            eval_diff_wf = None

            if eval_res_path is not None:
                eval_res_wf = codecs.open(eval_res_path, "w", "utf-8")
                wf_list.append(eval_res_wf)
            if eval_diff_path is not None:
                eval_diff_wf = codecs.open(eval_diff_path, "w", "utf-8")
                wf_list.append(eval_diff_wf)

            if eval_text_info is None:
                eval_text_info = ["\x01".join(x) for x in eval_feature]

            pred_label = list()
            for cur_pred_res, cur_eval_label, cur_text in \
                    zip(pred_res, eval_label, eval_text_info):
                cur_label = default_label
                cur_rate = "0"
                cur_evidence = "NULL"
                if len(cur_pred_res) > 0:
                    cur_label, cur_rate, cur_evidence = cur_pred_res[0]
                cur_label = label_trans_func(cur_label)
                pred_label.append(cur_label)
                if len(wf_list) > 0:
                    record = "\t".join([
                        cur_label,
                        cur_rate,
                        cur_eval_label,
                        cur_text,
                        cur_evidence,
                    ]) + "\n"

                    if eval_res_path is not None:
                        eval_res_wf.write(record)
                    if eval_diff_path is not None and cur_label != cur_eval_label:
                        eval_diff_wf.write(record)

            print(classification_report(eval_label, pred_label, digits=4))
        finally:
            for wf in wf_list:
                wf.close()

    def _single_predict(self, features, min_conf=0.05, digits=4, evidence=False, topk=10):
        """���������б�Ԥ����
        [in]  features : list[str], �����б�
              min_conf : float, ��Ϊ��ǩ����С���Ŷ�
              digits: int, ���ȿ����ڼ�λС��
              evidence: bool, true��෵��һ������������Ϣ �̶�����
              topk: int, ȡ����ֵǰtopk��������Ϊ֤��
        [out] pred_list: list[(str, str)], Ԥ������Ԫ���б�, (���, ���Ŷ�) �ɴ�С
        **����֤ ��liblinearģ���������һ�� ��ʱbiasΪ-1����Ĭ��ֵ��
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
                    # ��softmax�Ľ�����жϸ������Ե�ǰ�����Ҫ�� ����֤��ʱҪ��ʵ�ʸ������Ը����ֵ
                    evidence_dict[self.label_list[index]].append((feature, softmax_weights[index], weight))
                except IndexError as e:
                    print("index error. index = %d." % index)
                    raise e
    
        #logging.debug("label_weght_sum: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value)
        #                                                for label,value in label_value.items() ]))
    
        for label, sum_weight in label_value.items():
            label_value[label] = 1.0 / (1.0 + math.exp(-sum_weight))
            total += label_value[label]
    
        #logging.debug("label_value: %s" % ','.join(["[%s,%.2f]" % (label.encode("gb18030"), value)
        #                                            for label,value in label_value.items() ]))
        digits_format = "%%.%df" % digits
        pred_list = list()
        for label, weight_pred in sorted(label_value.items(), key=lambda x: x[1], reverse=True):
            pred_proba = weight_pred / total
            if pred_proba < min_conf:
                continue
            # ׼��֤��
            cur_label_evidence = "||".join(["%s(%.6f)" % (x[0], x[2]) for x in \
                    sorted(evidence_dict[label], key=lambda x:abs(x[1]), reverse=True)[:topk]])
            pred_list.append((label, digits_format % pred_proba, cur_label_evidence))
        return pred_list


if __name__ == "__main__":
    pass