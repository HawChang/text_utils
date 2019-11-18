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
    """给定数据集地址 给出数据的向量矩阵
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
        """初始化预处理类
        [in]  seg_method: str, 切词类型
              stopword_path: str, 停用词文件地址
              vec_method: str, 向量化方式
              feature_keep_percent: float, 保留特征的比例
              feature_keep_num: int, 保留特征的数目
              is_percent: bool, 保留特征的方式，按比例还是数目
              test_ratio: float, 验证集占总数据集的比例
              re_seg: bool, 是否重新从原始数据生成全部数据的特征信息
              data_root: str, 中间数据根目录
              model_root: str, 模型根目录
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

        # 中间数据地址
        self.total_data_path = data_root + "/total_data.txt"
        self.train_data_path = data_root + "/train_data.txt"
        self.val_data_path = data_root + "/test_data.txt"
        self.total_feature_path = data_root + "/total_feature.txt"
        self.train_feature_path = data_root + "/train_feature.txt"
        self.val_feature_path = data_root + "/test_feature.txt"
        self.train_lib_format_path = data_root + "/train_lib_format.txt"
        self.val_lib_format_path = data_root + "/test_lib_format.txt"

        # 模型地址
        self.reserved_feature_path = model_root + "/feature_id.txt"
        self.label_encoder_path = model_root + "/label_encoder.pkl"
        self.xgb_model_path = model_root + "/model"
        self.feature_weight_path = model_root + "/feature_weight.txt"
        log.info("Processor init succeed")

    def gen_data_vec(self,
            data_path,
            split_train_test=False,
            feature_select=False):
        """根据给定数据集地址 生成特征
        [in]  data_path: str, 数据集地址
              split_train_test: bool, true则划分测试集训练集
              feature_select: bool, true则进行特征选择
        [out] train_feature_vec: matrix, 训练数据特征矩阵
              train_label: list[str], 训练数据标签列表
              val_feature_vec: matrix, 验证数据特征矩阵, 若不划分训练验证数据集则为None
              val_label: list[str], 验证数据标签列表, 若不划分训练验证数据集则为None
        """
        # 首先根据数据集 生成数据的特征
        if self.re_seg:
            log.info("gen data feature.")
            start_time = time.time()
            data_list = get_data(data_path)
            # 调用函数 生成特征
            feature_list = [self.feature_label_gen(x) for x in data_list]
            # 存储数据信息
            write_to_file(data_list, self.total_data_path)
            # 存储特征信息
            write_to_file(feature_list, self.total_feature_path, \
                    write_func=lambda x : "%d\t%s" % x)
            log.info("cost_time : %.4f" % (time.time() - start_time))
        else:
            log.info("load data feature.")
            start_time = time.time()
            # 加载已有的数据和特征列表
            data_list = read_from_file(self.total_data_path)
            feature_list = read_from_file(self.total_feature_path, \
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
        write_to_file(train_data_list, self.train_data_path)
        write_to_file(train_feature_list, self.train_feature_path, \
                write_func=lambda x : "%s\t%s" % (x[0], x[1]))
        
        if split_train_test:
            write_to_file(val_data_list, self.val_data_path)
            write_to_file(val_feature_list, self.val_feature_path, \
                    write_func=lambda x : "%s\t%s" % (x[0], x[1]))

        # 特征列表中每个元素都是标签、特征的二元组
        train_label, train_feature = zip(*train_feature_list)
        if split_train_test:
            val_label, val_feature = zip(*val_feature_list)

        vectorizer = init_vectorizer(vec_method=self.vec_method)
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
                    reserved_feature_file=self.reserved_feature_path)
            
            # 生成各特征对应的id 从1开始 reserved_feature_name的顺序和reserved_feature_file中的顺序是一致的
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
        """根据字符串 提取其类别、特征 组成二元组
        [in]  line: str, 数据集每一行的内容
        [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
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
            """根据字符串 提取其类别、特征 组成二元组
            [in]  line: str, 数据集每一行的内容
            [out] res: (int, str), 二元组由类别和特征组成 特征由空格连接为字符串
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
