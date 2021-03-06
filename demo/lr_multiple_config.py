#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: config.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/21 20:53:39
"""

# 训练参数
re_seg = True
split_train_test = True
feature_select = False 
libsvm_format=False


# 训练集验证集参数
test_ratio = 0.2

# 特征生成参数
duplicate = False
idea_word_feature_sep = False
seg_method = "word_seg"
stopword_path = "./src/text_utils/dict/stopword.txt"
segdict_path = "./src/text_utils/dict/chinese_gbk"
ngram = 3
feature_min_length = 2


# 特征选择参数
vec_method = "count"
feature_keep_percent = 90
feature_keep_num = 10
is_percent = True
min_df = 3
rex = u"{关键词}|{投放地域}|{地域}|{|}"


to_file = True
# 模型训练时各数据地址
liblinear_train_path = "/home/work/zhanghao55/tools/liblinear-2.20/train"
train_data_dir = "./data/origin_data/multilabel_mark_data_processed"
mid_data_dir = "./local_data/"
model_dir = "./model/"
output_dir = "./output/"

