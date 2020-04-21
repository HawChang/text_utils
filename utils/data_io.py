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


import codecs
import logging
import os
import pickle
import time

from sklearn.datasets import dump_svmlight_file


def get_data(data_path, read_func=lambda x:x, encoding="gb18030", verbose=False):
    """��ȡ���ļ�(��Ŀ¼�������ļ�)������
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] data : list[str], ���ļ�(��Ŀ¼�������ļ�)������
    """
    file_list = get_file_name_list(data_path, verbose)
    data = list()
    for file_path in file_list:
        data.extend(read_from_file(file_path, read_func, encoding))
    return data


def get_file_name_list(data_path, verbose=False):
    """���ɹ������ݼ����ļ��б�
        ������ݼ���ַ���ļ����򷵻��б���ֻ�и��ļ���ַ
        ������ݼ���ַ��Ŀ¼���򷵻��б��а�����Ŀ¼�������ļ�����(����'.'��ͷ���ļ�)
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] file_list : list[str], ���ݼ��ļ������б�
    """
    from collections import deque
    file_list = list()
    path_stack = deque()
    path_stack.append(data_path)
    while len(path_stack) != 0:
        cur_path = path_stack.pop()
        logging.debug("check data path: %s." % cur_path)
        # ���ȼ��ԭʼ�������ļ������ļ���
        if os.path.isdir(cur_path):
            #logging.debug("data path is directory.")
            files = os.listdir(cur_path)
            # �����ļ����е�ÿһ���ļ�
            for file_name in files:
                # ����ļ�����.��ͷ ˵�����ļ����� ���������������ļ�
                if len(file_name) == 0 or file_name[0] == ".":
                    continue
                file_path = os.path.join(cur_path, file_name)
                path_stack.append(file_path)
        elif os.path.isfile(cur_path):
            #logging.info("data path is file. add to list.")
            file_list.append(cur_path)
        else:
            raise TypeError("unknown type of data path : %s" % cur_path)

    logging.info("file list top 20:")
    for index, file_name in enumerate(file_list[:20]):
        logging.info("#%d: %s" % (index + 1, file_name))
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
    """���ı��б����Ŀ���ļ���ַ
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
        logging.warning("pkl_path(\"%s\") illegal." % pkl_path)
    elif os.path.exists(pkl_path) and not overwrite:
        logging.warning("pkl_path(\"%s\") already exist and not over write." % pkl_path)
    else:
        with open(pkl_path, 'wb') as wf:
            pickle.dump(obj, wf)
        logging.debug("save to \"%s\" succeed." % pkl_path)


def label_encoder_save_as_class_id(label_encoder, class_id_path, conf_thres = 0.5):
    """��LabelEncoder����תΪdef-user�е�class_id.txt��ʽ����ʽ����ָ���ļ�
    [in]  label_encoder: class, ����
          class_id_path: str, �洢�ļ���ַ
          conf_thres: float, ������ֵ ����ֻ��ͳһ����
    """
    class_id_list = ["%d\t%s\t%f" % (index, str(class_name), conf_thres) for \
            index, class_name in enumerate(label_encoder.classes_)]
    write_to_file(class_id_list, class_id_path)
    logging.debug("trans label_encoder to \"%s\" succeed." % class_id_path)


def dump_libsvm_file(X, y, file_path, zero_based=False):
    """�����ݼ�תΪlibsvm��ʽ liblinear��xgboost��lightgbm�����Խ��ոø�ʽ
    [in]  X: array-like��sparse matrix, ��������
          y: array-like��sparse matrix, �����
          file_path: string��file-like in binary model, �ļ���ַ�����߶�������ʽ�򿪵Ŀ�д�ļ�
          zero_based: bool, true������id��0��ʼ liblinearѵ��ʱҪ������id��1��ʼ ���һ����ҪΪFalse
    """
    logging.debug("trans libsvm format data to %s." % file_path)
    start_time = time.time()
    dump_svmlight_file(X, y, file_path, zero_based=zero_based)
    logging.info("cost_time : %.4fs" % (time.time() - start_time))

if __name__ == "__main__":
    get_file_name_list("jinyong")
