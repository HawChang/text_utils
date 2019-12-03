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
reload(sys)
sys.setdefaultencoding("gb18030")

import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from utils.word_segger import WordSegger
from utils.data_io import read_from_file
from utils.data_io import dump_pkl
from utils.data_io import load_pkl
from utils.logger import Logger

log = Logger().get_logger()

class FeatureGenerator(object):
    def __init__(self,
            seg_method="word_seg",
            segdict_path="src/text_utils/dict/chinese_gbk",
            stopword_path=None,
            encoding="gb18030",
            ngram=3,
            feature_min_length=2):
        """
        ��ʼ������������
        [in] seg_method: str, ָ���дʷ�ʽ
             segdict_path: str, �ڲ��дʹ�����Ҫָ���д��ֵ��ļ���ַ
             stopword_path: str, ͣ�ô��ļ���ַ��ΪNone����ͣ�ô�
             encoding: str, ͣ�ô��ļ������ʽ
             n_gram: str: int, ngram����
             feature_min_length: int, ������С����
        """
        # �����������Ϣ
        self.seg_method = seg_method
        self.segdict_path = segdict_path
        self.stopword_path = stopword_path
        self.encoding = encoding
        self._ngram = 3 if ngram is None or ngram < 0 else ngram
        self._feature_min_length = 0 if feature_min_length is None else feature_min_length

        # �����дʹ��ߺ�ͣ�ô���Ϣ
        self._segger = WordSegger(self.seg_method, self.segdict_path)
        self._stopwords = set() if stopword_path is None \
                else set(read_from_file(stopword_path, encoding=encoding))


    def seg_words(self, text, verbose=False):
        """�д�
        [in] text: str, ���д��ַ���, unicode��gb18030����
        [out] valid_tokens: list[str], �дʽ��, unicode����
        """
        # �õ��дʽ�� �дʽ��Ϊ�ַ����б� unicode����
        tokens = self._segger.seg_words(text)
        if verbose:
            log.debug("tar string   : %s" % text.encode("gb18030"))
            log.debug("seg result   : %s" % "/ ".join(tokens).encode("gb18030"))
        # ȥ��ͣ�ô�
        valid_tokens = [x for x in tokens if len(x.strip()) != 0 and x not in self._stopwords]
        if verbose:
            log.debug("valid tokens : %s" % "/ ".join(valid_tokens).encode("gb18030"))
        return valid_tokens

    def gen_ngram_feature(self, token_list, duplicate=False):
        """�����б�����ngram����
        [in] token_list: list[str], �����б�, unicode����
             duplicate: bool, true�����ɵ�������ȥ��
        [out] features: list[str], �����б�, unicode����
        """
        feature_list = list()
        # ����ngram����
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
        """�����ַ�������ngram����
        [in] text: str, �ַ���, unicode��gb18030����
             verbose: bool, true����ʾִ��ϸ����Ϣ
             duplicate: bool, true������������������ȥ��
        [out] features: list[str], �����б�, unicode����
        """
        valid_tokens = self.seg_words(text, verbose)
        return self.gen_ngram_feature(valid_tokens, duplicate)
    
    def destroy(self):
        """�ͷ��ڴ�
        """
        self._segger.destroy()

    @staticmethod
    def save(generator, pkl_path, overwrite=False):
        """��������������
        """
        temp = generator._segger
        generator._segger = None
        dump_pkl(generator, pkl_path, overwrite)
        generator._segger = temp
    
    @staticmethod
    def load(pkl_path):
        """��������������
        """
        generator = load_pkl(pkl_path)
        generator._segger = WordSegger(generator.seg_method, generator.segdict_path)
        return generator


if __name__ == "__main__":
    generator = FeatureGenerator("word_seg", stopword_path="src/text_utils/dict/stopword.txt")
    #generator = FeatureGenerator("word_seg")
    
    tests = ["�����Ƿ��ܹ������д�",
             "����һ��]ͣ�ô�  ���߼�����,���ܹ������д�",
             u"��Ϸ���������,���ȶ�,����ƽ,���������������Ϸ ������ƽ̨ 24Сʱ����,\
             רҵ����,�����������Ϸ,ע�ἴ�ͺ������! ����ͬʱ����,�������ڴ����ļ���!��Ķ�Ŷ!"]

    for test in tests:
        feature_set = generator.gen_feature(test)
        print("/ ".join(feature_set).encode("gb18030"))

    FeatureGenerator.save(generator, "src/text_utils/test/output/generator.pkl", True)
    generator = FeatureGenerator.load("src/text_utils/test/output/generator.pkl")
    generator.destroy()
