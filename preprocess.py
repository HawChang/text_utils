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
from feature.feature_selector import FeatureSelector
from feature.feature_vectorizer import init_vectorizer

log = Logger().get_logger()
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

class ProcessFilePath(object):
    def __init__(self, output_dir="local_data"):
        """预处理流程中各输出文件地址
        [in] output_dir: str, 输出结果目录地址
        """
        self.output_dir = output_dir

        # 中间数据地址
        self.total_data_path = self.output_dir + "/total_data.txt"
        self.train_data_path = self.output_dir + "/train_data.txt"
        self.val_data_path = self.output_dir + "/test_data.txt"

        self.total_feature_path = self.output_dir + "/total_feature.txt"
        self.train_feature_path = self.output_dir + "/train_feature.txt"
        self.val_feature_path = self.output_dir + "/test_feature.txt"

        self.train_lib_format_path = self.output_dir + "/train_lib_format.txt"
        self.val_lib_format_path = self.output_dir + "/test_lib_format.txt"


class Preprocessor(object):
    """给定数据集地址 给出数据的向量矩阵
    """
    def __init__(self,
            feature_gen_func,
            vec_method="count",
            feature_keep_percent=90,
            feature_keep_num=10,
            is_percent=True,
            test_ratio=0.2,
            min_df=2):
        """初始化预处理类
        [in]  feature_gen_func: function, 生成特征的函数
              vec_method: str, 向量化方式
              feature_keep_percent: float, 保留特征的比例
              feature_keep_num: int, 保留特征的数目
              is_percent: bool, 保留特征的方式，按比例还是数目
              test_ratio: float, 验证集占总数据集的比例
              min_df: int, 特征保留的最小频次
        """
        self.feature_selector = FeatureSelector(
                feature_keep_percent=feature_keep_percent,
                feature_keep_num=feature_keep_num,
                is_percent=is_percent)
        self.feature_gen_func = feature_gen_func
        self.vec_method = vec_method
        self.test_ratio = test_ratio
        self.min_df = min_df

        log.info("Preprocessor init succeed")

    def gen_data_vec(self,
            data_path,
            feature_path=None,
            split_train_test=False,
            feature_select=False,
            to_file=True,
            re_seg=True,
            process_file_path=None):
        """根据给定数据集地址 生成特征
        [in]  data_path: str, 数据集地址
              feature_path: str, 特征保存地址
              split_train_test: bool, true则划分测试集训练集
              feature_select: bool, true则进行特征选择
              to_file: bool, true则中间数据会被存储
              re_seg: bool, 是否重新从原始数据生成全部数据的特征信息
              process_file_path: ProcessFilePath, 预处理时各文件地址
        [out] train_feature_vec: matrix, 训练数据特征矩阵
              train_label: list[str], 训练数据标签列表
              val_feature_vec: matrix, 验证数据特征矩阵, 若不划分训练验证数据集则为None
              val_label: list[str], 验证数据标签列表, 若不划分训练验证数据集则为None
        """
        # 如果要输出文件 则需要process_file_path
        if to_file and process_file_path is None:
            raise ValueError("specify process_file_path when output preprocess data")

        # 首先根据数据集 生成数据的特征
        if re_seg:
            log.info("gen data feature.")
            start_time = time.time()
            data_list = get_data(data_path)
            # 调用函数 生成特征
            feature_list = [self.feature_gen_func(x) for x in data_list]
            if to_file:
                # 存储数据信息
                write_to_file(data_list, process_file_path.total_data_path)
                # 存储特征信息
                write_to_file(feature_list, process_file_path.total_feature_path, \
                        write_func=lambda x : "%d\t%s" % x)
            log.info("cost_time : %.4f" % (time.time() - start_time))
        else:
            log.info("load data feature.")
            start_time = time.time()
            # 加载已有的数据和特征列表
            data_list = read_from_file(process_file_path.total_data_path)
            feature_list = read_from_file(process_file_path.total_feature_path, \
                    read_func=lambda x: x.strip("\n").split("\t"))
            feature_list = [(int(x[0]), x[1]) for x in feature_list]
            log.info("cost_time : %.4f" % (time.time() - start_time))
        
        if split_train_test:
            # 划分训练集、验证集
            train_data_list, val_data_list, train_feature_list, val_feature_list = \
                    train_test_split(data_list, feature_list, test_size=self.test_ratio, shuffle=True)
        else:
            train_data_list = data_list
            train_feature_list = feature_list

        # 存储数据和特征信息
        if to_file:
            write_to_file(train_data_list, process_file_path.train_data_path)
            write_to_file(train_feature_list, process_file_path.train_feature_path, \
                    write_func=lambda x : "%s\t%s" % (x[0], x[1]))
            
            if split_train_test:
                write_to_file(val_data_list, process_file_path.val_data_path)
                write_to_file(val_feature_list, process_file_path.val_feature_path, \
                        write_func=lambda x : "%s\t%s" % (x[0], x[1]))

        # 特征列表中每个元素都是标签、特征的二元组
        train_label, train_feature = zip(*train_feature_list)
        if split_train_test:
            val_label, val_feature = zip(*val_feature_list)

        vectorizer = init_vectorizer(vec_method=self.vec_method, min_df=self.min_df)
        # 构造数据特征向量
        train_feature_vec = vectorizer.fit_transform(train_feature)
        if split_train_test:
            val_feature_vec = vectorizer.transform(val_feature)

        # 筛选特征
        if feature_select:
            reserved_feature_name = self.feature_selector.fit(
                    train_feature_vec,
                    train_label,
                    vectorizer.get_feature_names(),
                    reserved_feature_file=feature_path)
            
            # 生成各特征对应的id 从1开始 reserved_feature_name的顺序和feature_path中的顺序是一致的
            feature_id_dict = {v:(ind) for ind, v in enumerate(reserved_feature_name)}

            vectorizer = init_vectorizer(vec_method=self.vec_method,vocabulary=feature_id_dict)
            train_feature_vec = vectorizer.transform(train_feature)
            if split_train_test:
                val_feature_vec = vectorizer.transform(val_feature)
        elif to_file:
            write_to_file([(ind+1, x) for ind, x in enumerate(vectorizer.get_feature_names())],
                    feature_path, write_func=lambda x: "%d\t%s" % x)
        log.info("train feature vec shape: %s." % str(train_feature_vec.shape))
        if split_train_test:
            log.info("test feature vec shape: %s." % str(val_feature_vec.shape))
        
        if to_file:
            log.info("trans to libsvm data file.")
            start_time = time.time()
            dump_libsvm_file(train_feature_vec, train_label, process_file_path.train_lib_format_path)
            if split_train_test:
                dump_libsvm_file(val_feature_vec, val_label, process_file_path.val_lib_format_path)
            log.info("cost_time : %.4f" % (time.time() - start_time))
        
        if not split_train_test:
            val_feature_vec = None
            val_label = None

        return  train_feature_vec, train_label, val_feature_vec, val_label


if __name__ == "__main__":
    from utils.for_def_user import LabelEncoder
    from feature.feature_generator import FeatureGenerator

    process_file_path = ProcessFilePath(output_dir="test/output/")
    label_encoder = LabelEncoder("test/class_id.txt")
    feature_generator = FeatureGenerator(
            seg_method="word_seg",
            segdict_path="dict/chinese_gbk",
            stopword_path="dict/stopword.txt",
            ngram=3,
            feature_min_length=1)
    duplicate = False
    line_process_num = 0

    def feature_label_gen(line):
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
        """
        global line_process_num
        parts = line.strip("\n").split("\t")
        label = label_encoder.transform(parts[0])
        idea_list = parts[1].split("||")
        word_list = parts[2].split("||")
        feature_list = list()
        line_process_num += 1
        for text in idea_list + word_list:
            feature_list.extend(feature_generator.gen_feature(text, duplicate=duplicate))
        if line_process_num % 4000 == 0:
            text = "||".join(parts[1:3])
            seg_text = "/ ".join(feature_generator.seg_words(text))
            log.debug("process line num #%d" % line_process_num)
            log.debug("origin  : %s" % text.encode("gb18030"))
            log.debug("="*150)
            log.debug("seg res : %s" % seg_text.encode("gb18030"))
        features = feature_list if duplicate else set(feature_list)
        return (label, " ".join(features))
    

    test_processor = Preprocessor(
            feature_gen_func=feature_label_gen,
            vec_method="count",
            feature_keep_percent=40,
            feature_keep_num=10,
            is_percent=True,
            test_ratio=0.2,
            min_df=2)

    test_processor.gen_data_vec("test/train_data.txt",
            "test/output/feature_id.txt",
            split_train_test=False,
            feature_select=False,
            to_file=True,
            re_seg=True,
            process_file_path=process_file_path)

    test_processor.gen_data_vec("test/train_data.txt",
            "test/output/feature_id.txt",
            split_train_test=False,
            feature_select=True,
            to_file=True,
            re_seg=True,
            process_file_path=process_file_path)

    test_processor.gen_data_vec("test/train_data.txt",
            "test/output/feature_id.txt",
            split_train_test=True,
            feature_select=True,
            to_file=True,
            re_seg=True,
            process_file_path=process_file_path)
