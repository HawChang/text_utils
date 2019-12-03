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
feature_select = True


# 训练集验证集参数
test_ratio = 0.2

# 特征生成参数
duplicate = False
idea_word_feature_sep = False
seg_method = "word_seg"
stopword_path = "./src/text_utils/dict/stopword.txt"
ngram = 3
feature_min_length = 2


# 特征选择参数
vec_method = "count"
feature_keep_percent = 90
feature_keep_num = 10
is_percent = True
min_df = 3


# 模型训练时各数据地址
train_data_root = "../mark_data/train_data/"
data_root = "./local_data/"
model_root = "./model/"

