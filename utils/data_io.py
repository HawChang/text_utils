#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_io.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/19 20:44:30
"""

import sys
reload(sys)
sys.setdefaultencoding("gb18030")

import codecs
import os
import pickle
import time

from sklearn.datasets import dump_svmlight_file

from logger import Logger

log = Logger().get_logger()


def get_data(data_path, read_func=lambda x:x, encoding="gb18030", verbose=False):
    """��ȡ���ļ�(��Ŀ¼�������ļ�)������
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] data : list[str], ���ļ�(��Ŀ¼�������ļ�)������
    """
    file_list = get_file_name_list(data_path, verbose)
    data = list()
    for file_path in file_list:
        data.extend(read_from_file(file_path, read_func))
    return data


def get_file_name_list(data_path, verbose=False):
    """���ɹ������ݼ����ļ��б�
        ������ݼ���ַ���ļ����򷵻��б���ֻ�и��ļ���ַ
        ������ݼ���ַ��Ŀ¼���򷵻��б��а�����Ŀ¼�������ļ�����(����'.'��ͷ���ļ�)
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] file_list : list[str], ���ݼ��ļ������б�
    """
    file_list = list()
    log.info("check data path: %s." % data_path)
    # ���ȼ��ԭʼ�������ļ������ļ���
    if os.path.isdir(data_path):
        log.info("data path is directory.")
        files = os.listdir(data_path)
        # �����ļ����е�ÿһ���ļ�
        for file_name in files:
            # ����ļ�����.��ͷ ˵�����ļ����� ���������������ļ�
            if len(file_name) == 0 or file_name[0] == ".":
                continue
            file_path = os.path.join(data_path, file_name)
            file_list.append(file_path)
    elif os.path.isfile(data_path):
        log.info("data path is file.")
        file_list.append(data_path)
    else:
        raise TypeError("unknown type of data path : %s." % data_path)

    log.info("file list:")
    for index, file_name in enumerate(file_list):
        log.info("#%d: %s" % (index + 1, file_name))
    return file_list


def read_from_file(file_path, read_func=lambda x:x, encoding="gb18030"):
    """�����ļ��еĴ�
    [in] file_path: str, �ļ���ַ
    [out] word_list: list[str], �����б�
    """
    word_list = list()
    with codecs.open(file_path, "r", encoding) as rf:
        for line in rf:
            word_list.append(read_func(line.strip("\n")))
    return word_list


def write_to_file(text_list, dst_file_path, write_func=lambda x:x, encoding="gb18030"):
    """���ı��б�����Ŀ���ļ���ַ
    [in]  text_list: list[str], �ı��б�
          dst_file_path: str, Ŀ���ļ���ַ
    """
    with codecs.open(dst_file_path, "w", encoding) as wf:
        wf.write("\n".join([write_func(x) for x in text_list]))


def load_pkl(pkl_path):
    """���ض���
    [in]  pkl_path: str, �����ļ���ַ
    [out] obj: class, ����
    """
    with open(pkl_path, 'rb') as rf:
        return pickle.load(rf)


def dump_pkl(obj, pkl_path, overwrite=False):
    """�洢����
    [in]  obj: class, ����
          pkl_path: str, �����ļ���ַ
          overwrite: bool, �Ƿ񸲸ǣ�False���ļ�����ʱ���洢
    """
    if len(pkl_path) == 0 or pkl_path is None:
        log.warning("pkl_path(\"%s\") illegal." % pkl_path)
    elif os.path.exists(pkl_path) and not overwrite:
        log.warning("pkl_path(\"%s\") already exist and not over write." % pkl_path)
    else:
        with open(pkl_path, 'wb') as wf:
            pickle.dump(obj, wf)
        log.debug("save to \"%s\" succeed." % pkl_path)


def label_encoder_save_as_class_id(label_encoder, class_id_path, conf_thres = 0.5):
    """��LabelEncoder����תΪdef-user�е�class_id.txt��ʽ����ʽ����ָ���ļ�
    [in]  label_encoder: class, ����
          class_id_path: str, �洢�ļ���ַ
          conf_thres: float, ������ֵ ����ֻ��ͳһ����
    """
    class_id_list = ["%d\t%s\t%f" % (index, str(class_name), conf_thres) for \
            index, class_name in enumerate(label_encoder.classes_)]
    write_to_file(class_id_list, class_id_path)
    log.debug("trans label_encoder to \"%s\" succeed." % class_id_path)


def dump_libsvm_file(X, y, file_path, zero_based=False):
    """�����ݼ�תΪlibsvm��ʽ liblinear��xgboost��lightgbm�����Խ��ոø�ʽ
    [in]  X: array-like��sparse matrix, ��������
          y: array-like��sparse matrix, �����
          file_path: string��file-like in binary model, �ļ���ַ�����߶�������ʽ�򿪵Ŀ�д�ļ�
          zero_based: bool, true������id��0��ʼ liblinearѵ��ʱҪ������id��1��ʼ ���һ����ҪΪFalse
    """
    log.debug("trans libsvm format data to %s." % file_path)
    start_time = time.time()
    dump_svmlight_file(X, y, file_path, zero_based=zero_based)
    log.info("cost_time : %.4fs" % (time.time() - start_time))