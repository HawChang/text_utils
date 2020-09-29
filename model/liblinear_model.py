#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: liblinear_model.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/11/04 16:22:30
"""

import codecs
import os
import subprocess
import sys
import time
import warnings

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from utils.logger import Logger

log = Logger().get_logger()
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def argsort(number_list):
    """排序由小到大，但返回的是下标排序结果
    例如：
        数组a=[3,5,4,2,1] 则argsort(a)=[4,3,0,2,1]：表示第4位是a数组中值最小的,第3位是a数组中第二小的
    [in] number_list: list[可直接比较大小的元素], 待排序列表
    """
    return sorted(range(len(number_list)), key=lambda x:number_list[x])


def liblinear_train(
        train_data_path,
        model_path,
        liblinear_train_path="/home/work/zhanghao55/tools/liblinear-2.20/train",
        model_conf="-s 0",
        feature_weight_save_path=None,
        feature_name_list=None):
    """训练模型
    [in] train_data_path: str, 训练数据地址
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
        log.info("liblinear train start...")
        start_time = time.time()
        cmd_str =' '.join([
                liblinear_train_path,
                model_conf,
                train_data_path,
                model_path])
        log.debug("cmd str: %s" % cmd_str)
        popen = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1)

        # 重定向标准输出
        while popen.poll() is None:
            # None表示正在执行中
            r = popen.stdout.readline().strip("\n")
            log.debug(r)

        # 重定向错误输出
        if popen.poll() != 0:
            # 不为0表示执行错误
            err = popen.stderr.read().strip("\n")
            log.error(err)

        log.info("cost time %.4fs." % (time.time() - start_time))
    except subprocess.CalledProcessError as e:
        log.warning("liblinear train failed.")
        log.error(e)
        raise e

    if feature_weight_save_path is not None:
        log.info("trans liblinear model to feature_weight_file")
        if feature_name_list is None:
            log.warning("feature_weight_file relies on feature_name_list to creat. exit.")
        else:
            gen_feature_weight_file(model_path, feature_name_list, feature_weight_save_path)

    log.info("liblinear train finish")


def liblinear_format(feature_list, feature_id_dict):
    """将特征列表转换成liblinear格式的字符串
    [in]  feature_list: list[string], 特征列表
          feature_id_dict: dict{string:int}, 特征对应的id字典
    [out] str, 特征列表转成liblinear格式的字符串结果
    """
    feature_ids = set()
    for feature in feature_list:
        if feature in feature_id_dict:
            feature_ids.add(feature_id_dict[feature])

    feature_ids = sorted(feature_ids, reverse=False)
    return '\t'.join(map(lambda x: "%d:1" % x, feature_ids))


def gen_feature_weight_file(liblinear_model_path, feature_name_list, feature_weight_save_path):
    """根据liblinear生成的模型文件和特征保留文件生成通用的multiclass特征权重文件
    [in]  liblinear_model_path: str, liblinear模型文件地址
          feature_name_list: list[str], 特征字面列表 其顺序应与liblinear模型文件中特征的顺序一致
          feature_weight_save_path: str, 特征权重文件地址
    """
    # 读入模型文件中的特征权值
    # 按类别值从小到大的顺序给出
    class_num = None
    feature_index = 0
    log.debug("gen_feature_weight_file start...")
    start_time = time.time()
    with codecs.open(liblinear_model_path, "r", "gb18030") as rf, \
            codecs.open(feature_weight_save_path, "w", "gb18030") as wf:
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
                log.info("position rank: " + ",".join([str(x) for x in index_transfer]))
                label_list = list()
                for class_index in range(class_num):
                    label_list.append(str(labels[index_transfer[class_index]]))
                wf.write("classes:\t"+",".join(label_list)+"\n")

            if index < 6:
                continue

            # liblinear权值文件 最后会有一个空格
            weights = line.strip(" ").split(" ")
            assert len(weights) == class_num, "wrong weight num at line #%d, expect %d, actual %d." % (index+1, class_num, len(weights))
            reordered_weights = list()
            for weight_index in range(class_num):
                reordered_weights.append(weights[index_transfer[weight_index]])
            wf.write("\t".join([feature_name_list[feature_index], " ".join(reordered_weights)]) + "\n")
            feature_index += 1
    log.info("cost time %.4fs." % (time.time() - start_time))


if __name__ == "__main__":
    reserved_feature_path = "./lib_model/reserved_feature.txt"
    train_data_path = "./lib_data/train_feature.txt"
    train_lib_format_path = "./lib_data/train_lib_format.txt"
    lib_model_path = "./lib_model/lib_model.txt"
    feature_weight_save_path = "./lib_model/feature_weight.txt"
    label_encoder_path = "./lib_model/label_encoder.pkl"

    from utils.data_io import read_from_file
    from utils.data_io import load_pkl
    # 数组下标就是该feature的id
    feature_name_list = read_from_file(reserved_feature_path, \
            read_func=lambda x: x.strip("\n").split("\t")[1])
    # 字符串列表转为特征id字典 生成各特征对应的id 从1开始
    feature_id_dict = {v:(ind+1) for ind, v in enumerate(feature_name_list)}
    label_encoder = load_pkl(label_encoder_path)

    def trans_lib_format():
        log.debug("trans_lib_format start...")
        start_time = time.time()
        with codecs.open(train_data_path, "r", 'gb18030') as rf, \
                codecs.open(train_lib_format_path, "w", "gb18030") as wf:
            for line in rf:
                parts= line.strip("\n").split("\t")
                label = str(label_encoder.transform([parts[0]])[0])
                feature_list = parts[1].split(" ")
                lib_format_data = label + "\t" + liblinear_format(feature_list, feature_id_dict)
                wf.write(lib_format_data + "\n")
        log.debug("cost time %.4fs." % (time.time() - start_time))

    trans_lib_format()
    
    liblinear_train(
            train_lib_format_path,
            lib_model_path,
            feature_weight_save_path=feature_weight_save_path,
            feature_name_list=feature_name_list)

    #gen_feature_weight_file(lib_model_path, feature_name_list, feature_weight_save_path)
