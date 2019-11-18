#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: preprocess.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/19 22:04:21
"""

import sys
import time
import warnings

from sklearn.model_selection import train_test_split

from utils.logger import Logger
from utils.data_io import get_data
from utils.data_io import write_to_file
from utils.data_io import read_from_file
from utils.data_io import dump_libsvm_file
from feature.feature_generator import FeatureGenerator
from feature.feature_selector import FeatureSelector
from feature.feature_vectorizer import init_vectorizer

log = Logger().get_logger()
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

class Preprocessor(object):
    """�������ݼ���ַ �������ݵ���������
    """
    def __init__(self,
            seg_method="word_seg",
            stopword_path="data/stopword.txt",
            vec_method="count",
            feature_keep_percent=90,
            feature_keep_num=10,
            is_percent=True,
            test_ratio=0.2,
            min_df=2,
            re_seg=True,
            data_root="local_data",
            model_root="model"):
        """��ʼ��Ԥ������
        [in]  seg_method: str, �д�����
              stopword_path: str, ͣ�ô��ļ���ַ
              vec_method: str, ��������ʽ
              feature_keep_percent: float, ���������ı���
              feature_keep_num: int, ������������Ŀ
              is_percent: bool, ���������ķ�ʽ��������������Ŀ
              test_ratio: float, ��֤��ռ�����ݼ��ı���
              re_seg: bool, �Ƿ����´�ԭʼ��������ȫ�����ݵ�������Ϣ
              data_root: str, �м����ݸ�Ŀ¼
              model_root: str, ģ�͸�Ŀ¼
        """
        self.feature_generator = FeatureGenerator(
                seg_method,
                stopword_path=stopword_path) 
        self.feature_selector = FeatureSelector(
                feature_keep_percent=feature_keep_percent,
                feature_keep_num=feature_keep_num,
                is_percent=is_percent)
        self.vec_method = vec_method
        self.line_process_num = 0
        self.test_ratio = test_ratio
        self.min_df = min_df
        self.re_seg = re_seg

        # �м����ݵ�ַ
        self.total_data_path = data_root + "/total_data.txt"
        self.train_data_path = data_root + "/train_data.txt"
        self.val_data_path = data_root + "/test_data.txt"
        self.total_feature_path = data_root + "/total_feature.txt"
        self.train_feature_path = data_root + "/train_feature.txt"
        self.val_feature_path = data_root + "/test_feature.txt"
        self.train_lib_format_path = data_root + "/train_lib_format.txt"
        self.val_lib_format_path = data_root + "/test_lib_format.txt"

        # ģ�͵�ַ
        self.reserved_feature_path = model_root + "/feature_id.txt"
        self.label_encoder_path = model_root + "/label_encoder.pkl"
        self.xgb_model_path = model_root + "/model"
        self.feature_weight_path = model_root + "/feature_weight.txt"
        log.info("Processor init succeed")

    def gen_data_vec(self,
            data_path,
            split_train_test=False,
            feature_select=False):
        """���ݸ������ݼ���ַ ��������
        [in]  data_path: str, ���ݼ���ַ
              split_train_test: bool, true�򻮷ֲ��Լ�ѵ����
              feature_select: bool, true���������ѡ��
        [out] train_feature_vec: matrix, ѵ��������������
              train_label: list[str], ѵ�����ݱ�ǩ�б�
              val_feature_vec: matrix, ��֤������������, ��������ѵ����֤���ݼ���ΪNone
              val_label: list[str], ��֤���ݱ�ǩ�б�, ��������ѵ����֤���ݼ���ΪNone
        """
        # ���ȸ������ݼ� �������ݵ�����
        if self.re_seg:
            log.info("gen data feature.")
            start_time = time.time()
            data_list = get_data(data_path)
            # ���ú��� ��������
            feature_list = [self.feature_label_gen(x) for x in data_list]
            # �洢������Ϣ
            write_to_file(data_list, self.total_data_path)
            # �洢������Ϣ
            write_to_file(feature_list, self.total_feature_path, \
                    write_func=lambda x : "%d\t%s" % x)
            log.info("cost_time : %.4f" % (time.time() - start_time))
        else:
            log.info("load data feature.")
            start_time = time.time()
            # �������е����ݺ������б�
            data_list = read_from_file(self.total_data_path)
            feature_list = read_from_file(self.total_feature_path, \
                    read_func=lambda x: x.strip("\n").split("\t"))
            feature_list = [(int(x[0]), x[1]) for x in feature_list]
            log.info("cost_time : %.4f" % (time.time() - start_time))
        
        if split_train_test:
            # ����ѵ��������֤��
            train_data_list, val_data_list, train_feature_list, val_feature_list = \
                    train_test_split(data_list, feature_list, test_size=self.test_ratio, shuffle=True)
        else:
            train_data_list = data_list
            train_feature_list = feature_list

        # �洢���ݺ�������Ϣ
        write_to_file(train_data_list, self.train_data_path)
        write_to_file(train_feature_list, self.train_feature_path, \
                write_func=lambda x : "%s\t%s" % (x[0], x[1]))
        
        if split_train_test:
            write_to_file(val_data_list, self.val_data_path)
            write_to_file(val_feature_list, self.val_feature_path, \
                    write_func=lambda x : "%s\t%s" % (x[0], x[1]))

        # �����б���ÿ��Ԫ�ض��Ǳ�ǩ�������Ķ�Ԫ��
        train_label, train_feature = zip(*train_feature_list)
        if split_train_test:
            val_label, val_feature = zip(*val_feature_list)

        vectorizer = init_vectorizer(vec_method=self.vec_method)
        # ����������������
        train_feature_vec = vectorizer.fit_transform(train_feature)
        if split_train_test:
            val_feature_vec = vectorizer.transform(val_feature)

        # ɸѡ����
        if feature_select:
            reserved_feature_name = self.feature_selector.fit(
                    train_feature_vec,
                    train_label,
                    vectorizer.get_feature_names(),
                    reserved_feature_file=self.reserved_feature_path)
            
            # ���ɸ�������Ӧ��id ��1��ʼ reserved_feature_name��˳���reserved_feature_file�е�˳����һ�µ�
            feature_id_dict = {v:(ind) for ind, v in enumerate(reserved_feature_name)}

            vectorizer = init_vectorizer(vec_method=self.vec_method,vocabulary=feature_id_dict)
            train_feature_vec = vectorizer.transform(train_feature)
            if split_train_test:
                val_feature_vec = vectorizer.transform(val_feature)
        else:
            write_to_file([(ind+1, x) for ind, x in enumerate(vectorizer.get_feature_names())],
                    self.reserved_feature_path, write_func=lambda x: "%d\t%s" % x)
        log.info("train feature vec shape: %s." % str(train_feature_vec.shape))
        if split_train_test:
            log.info("test feature vec shape: %s." % str(val_feature_vec.shape))
        
        log.info("trans to libsvm data file.")
        start_time = time.time()
        dump_libsvm_file(train_feature_vec, train_label, self.train_lib_format_path)
        if split_train_test:
            dump_libsvm_file(val_feature_vec, val_label, self.val_lib_format_path)
        log.info("cost_time : %.4f" % (time.time() - start_time))
        
        if not split_train_test:
            val_feature_vec = None
            val_label = None

        return  train_feature_vec, train_label, val_feature_vec, val_label

    def feature_label_gen(self, line):
        """�����ַ��� ��ȡ��������� ��ɶ�Ԫ��
        [in]  line: str, ���ݼ�ÿһ�е�����
        [out] res: (int, str), ��Ԫ��������������� �����ɿո�����Ϊ�ַ���
        """
        raise NotImplementedError("feature label gen function should be implemented.")

if __name__ == "__main__":
    from utils.for_def_user import LabelEncoder
    #label_encoder = LabelEncoder("src/text_utils/test/class_id.txt")
    duplicate = False
    class TestProcessor(Preprocessor):
        def __init__(self, **kwargs):
            super(TestProcessor, self).__init__(**kwargs)
            self.label_encoder = LabelEncoder("src/text_utils/test/class_id.txt")
            log.info("TestProcessor init succeed")

        def feature_label_gen(self, line):
            """�����ַ��� ��ȡ��������� ��ɶ�Ԫ��
            [in]  line: str, ���ݼ�ÿһ�е�����
            [out] res: (int, str), ��Ԫ��������������� �����ɿո�����Ϊ�ַ���
            """
            parts = line.strip("\n").split("\t")
            label = self.label_encoder.transform(parts[0])
            idea_list = parts[1].split("||")
            word_list = parts[2].split("||")
            feature_list = list()
            self.line_process_num += 1
            for text in idea_list + word_list:
                feature_list.extend(self.feature_generator.gen_feature(text, duplicate=duplicate))
            if self.line_process_num % 4000 == 0:
                text = "||".join(parts[1:3])
                seg_text = "/ ".join(self.feature_generator.seg_words(text))
                log.debug("process line num #%d" % self.line_process_num)
                log.debug("origin  : %s" % text.encode("gb18030"))
                log.debug("="*150)
                log.debug("seg res : %s" % seg_text.encode("gb18030"))
            features = feature_list if duplicate else set(feature_list)
            return (label, " ".join(features))

    test_processor = TestProcessor(
            seg_method="word_seg",
            stopword_path="data/stopword.txt",
            vec_method="count",
            feature_keep_percent=40,
            feature_keep_num=10,
            is_percent=True,
            test_ratio=0.2,
            min_df=2,
            re_seg=True,
            data_root="src/text_utils/test/output/",
            model_root="src/text_utils/test/output/")

    test_processor.gen_data_vec("src/text_utils/test/train_data.txt",
            split_train_test=False,
            feature_select=False)

    test_processor.gen_data_vec("src/text_utils/test/train_data.txt",
            split_train_test=False,
            feature_select=True)
    test_processor.gen_data_vec("src/text_utils/test/train_data.txt",
            split_train_test=True,
            feature_select=True)
