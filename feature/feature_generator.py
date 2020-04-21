#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: fature_generator.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/20 16:00:37
"""

import sys
import logging
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from utils.word_segger import WordSegger
from utils.data_io import read_from_file
from utils.data_io import dump_pkl
from utils.data_io import load_pkl


class FeatureGenerator(object):
    def __init__(self,
            seg_method="word_seg",
            segdict_path="src/text_utils/dict/chinese_gbk",
            stopword_path=None,
            encoding="gb18030",
            ngram=3,
            feature_min_length=2):
        """
        初始化特征生成类
        [in] seg_method: str, 指定切词方式
             segdict_path: str, 内部切词工具需要指明切词字典文件地址
             stopword_path: str, 停用词文件地址，为None则无停用词
             encoding: str, 停用词文件编码格式
             n_gram: str: int, ngram参数
             feature_min_length: int, 特征最小长度
        """
        # 保存各配置信息
        self.seg_method = seg_method
        self.segdict_path = segdict_path
        self.stopword_path = stopword_path
        self.encoding = encoding
        self._ngram = 3 if ngram is None or ngram < 0 else ngram
        self._feature_min_length = 0 if feature_min_length is None else feature_min_length

        # 加载切词工具和停用词信息
        self._segger = WordSegger(self.seg_method, self.segdict_path)
        self._stopwords = set() if stopword_path is None \
                else set(read_from_file(stopword_path, encoding=encoding))


    def seg_words(self, text, verbose=False):
        """切词
        [in] text: str, 待切词字符串, unicode或gb18030编码
        [out] valid_tokens: list[str], 切词结果, unicode编码
        """
        # 得到切词结果 切词结果为字符串列表 unicode编码
        tokens = self._segger.seg_words(text)
        if verbose:
            logging.debug("tar string   : %s" % text.encode("gb18030"))
            logging.debug("seg result   : %s" % "/ ".join(tokens).encode("gb18030"))
        # 去除停用词
        valid_tokens = [x for x in tokens if len(x.strip()) != 0 and x not in self._stopwords]
        if verbose:
            logging.debug("valid tokens : %s" % "/ ".join(valid_tokens).encode("gb18030"))
        return valid_tokens

    def gen_ngram_feature(self, token_list, duplicate=False):
        """根据列表生成ngram特征
        [in] token_list: list[str], 单词列表, unicode编码
             duplicate: bool, true则生成的特征不去重
        [out] features: list[str], 特征列表, unicode编码
        """
        feature_list = list()
        # 生成ngram特征
        for start_pos in range(len(token_list)):
            cur_feature = ""
            for offset in range(min(len(token_list)-start_pos, self._ngram)):
                cur_feature += token_list[start_pos + offset]
                if len(cur_feature) >= self._feature_min_length:
                    feature_list.append(cur_feature)
        if duplicate:
            return feature_list
        else:
            return set(feature_list)

    def gen_feature(self, text, verbose=False, duplicate=False):
        """根据字符串生成ngram特征
        [in] text: str, 字符串, unicode或gb18030编码
             verbose: bool, true则显示执行细节信息
             duplicate: bool, true则生成所有特征，不去重
        [out] features: list[str], 特征列表, unicode编码
        """
        valid_tokens = self.seg_words(text, verbose)
        return self.gen_ngram_feature(valid_tokens, duplicate)
    
    def destroy(self):
        """释放内存
        """
        self._segger.destroy()

    @staticmethod
    def save(generator, pkl_path, overwrite=False):
        """保存特征生成类
        """
        temp = generator._segger
        generator._segger = None
        dump_pkl(generator, pkl_path, overwrite)
        generator._segger = temp
    
    @staticmethod
    def load(pkl_path):
        """加载特征生成类
        """
        generator = load_pkl(pkl_path)
        generator._segger = WordSegger(generator.seg_method, generator.segdict_path)
        return generator


if __name__ == "__main__":
    generator = FeatureGenerator("word_seg", stopword_path="src/text_utils/dict/stopword.txt")
    #generator = FeatureGenerator("word_seg")
    
    tests = ["测试是否能够正常切词",
             "测试一下]停用词  的逻辑，是,否？能够正常切词",
             u"游戏极多的棋牌,极稳定,极公平,人气极多的棋牌游戏 的棋牌平台 24小时服务,\
             专业贴心,处理的棋牌游戏,注册即送海量金币! 万人同时在线,的棋牌期待您的加入!你的懂哦!"]

    for test in tests:
        feature_set = generator.gen_feature(test)
        print("/ ".join(feature_set).encode("gb18030"))

    FeatureGenerator.save(generator, "src/text_utils/test/output/generator.pkl", True)
    generator = FeatureGenerator.load("src/text_utils/test/output/generator.pkl")
    generator.destroy()
