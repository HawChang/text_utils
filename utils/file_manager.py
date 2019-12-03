#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: file_manager.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/21 16:34:11
"""

from logger import Logger

log = Logger().get_logger()

class FileManager(object):
    def __init__(self, data_root="local_data", model_root="model", output_root="output"):
        """�����ļ���ַ��ʼ��
        [in]  data_root: str, �����ļ�Ŀ¼��ַ
              model_root: str, ģ���ļ�Ŀ¼��ַ
              output_root: str, ������Ŀ¼��ַ
        """
        # Ŀ¼��Ϣ
        self.data_root = data_root
        self.model_root = model_root

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
        self.class_id_path = model_root + "/class_id.txt"
        self.generator_path = model_root + "/generator.pkl"
        self.reserved_feature_path = model_root + "/feature_id.txt"
        self.model_path = model_root + "/model.txt"
        self.feature_weight_path = model_root + "/feature_weight.txt"

        # �������ļ���ַ
        self.pred_res_path = output_root + "/pred_res.txt"
        self.wrong_pred_res_path = output_root + "/wrong_pred_res.txt"

        log.debug("FileManager init succeed")
