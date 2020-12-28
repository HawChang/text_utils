#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   lr_tokenizer.py
Author:   zhanghao55@baidu.com
Date  :   20/09/28 14:22:12
Desc  :   
"""

import json
import logging
import os
import sys
import time

from text_utils.utils.data_io import read_from_file
from text_utils.utils.word_segger import WordSegger


class LRTokenizer():
    def __init__(self,
            seg_method="word_seg",
            segdict_path="dict/chinese_gbk",
            stopword_path=None,
            encoding="gb18030",
            lowercase=True,
            ngram=3,
            feature_min_length=2,
            jieba_tmp_dir=None,
            ):
        """
        ��ʼ��tokenizer
        [in] stopword_path: str, ͣ�ô��ļ���ַ��ΪNone����ͣ�ô�
             encoding: str, ͣ�ô��ļ�����
             lowercase: bool, �Ƿ�ͳһתΪСд
             n_gram: str: int, ngram����
             feature_min_length: int, ������С����
             jieba_tmp_dir: str, ��ͷִʵ���ʱ�ļ��е�ַ
        """
        # �����������Ϣ
        self._seg_method = seg_method
        self._segdict_path = segdict_path
        self._stopword_path = stopword_path
        self._encoding = encoding
        self._lowercase = lowercase
        self._ngram = 3 if ngram is None or ngram < 0 else ngram
        self._feature_min_length = 0 if feature_min_length is None else feature_min_length
        self._jieba_tmp_dir = jieba_tmp_dir

        self.config_dict = {
                "seg_method": self._seg_method,
                "segdict_path": self._segdict_path,
                "stopword_path": self._stopword_path,
                "encoding": self._encoding,
                "lowercase": self._lowercase,
                "ngram": self._ngram,
                "feature_min_length": self._feature_min_length,
                "jieba_tmp_dir": self._jieba_tmp_dir,
                }

        # �����дʹ��ߺ�ͣ�ô���Ϣ
        self._segger = WordSegger(self._seg_method, self._segdict_path, self._jieba_tmp_dir)
        self._stopwords = set() if stopword_path is None \
                else set(read_from_file(stopword_path, encoding=encoding))

    def seg_words(self, text, verbose=False):
        """�д�
        [in] text: str, ���д��ַ���, unicode��gb18030����
        [out] valid_tokens: list[str], �дʽ��, unicode����
        """
        # �õ��дʽ�� �дʽ��Ϊ�ַ��������� unicode����
        tokens = self._segger.seg_words(text)
        if verbose:
            logging.debug("tar string   : %s" % text.encode("gb18030"))
            logging.debug("seg result   : %s" % "/ ".join(tokens).encode("gb18030"))
        # ȥ��ͣ�ô�
        valid_tokens = [x for x in tokens if len(x.strip()) != 0 and x not in self._stopwords]
        if verbose:
            logging.debug("valid tokens : %s" % "/ ".join(valid_tokens).encode("gb18030"))
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

    def encode(self, text, verbose=False, duplicate=False):
        """�����ַ�������ngram����,����ernie_tokenizer��encode����
        [in] text: str, �ַ���, unicode��gb18030����
             verbose: bool, true����ʾִ��ϸ����Ϣ
             duplicate: bool, true������������������ȥ��
        [out] feature_list: list[list[str]], �����б���б�,ֻȡ��һ��Ԫ�ؼ���(����ernie_tokenizer��encode), unicode����
        """
        if self._lowercase:
            text = text.lower()
        valid_tokens = self.seg_words(text, verbose)
        features = self.gen_ngram_feature(valid_tokens, duplicate)
        return [features]

    def transform(self, text_list, verbose=False, duplicate=False):
        """�����ַ�������ngram����
        [in] text_list: list[str], �ַ����б�, unicode��gb18030����
             verbose: bool, true����ʾִ��ϸ����Ϣ
             duplicate: bool, true������������������ȥ��
        [out] feature_list: list[list[str]], �����б���б�, unicode����
        """
        feature_list = list()
        start_time = time.time()
        for text in text_list:
            if self._lowercase:
                text = text.lower()
            valid_tokens = self.seg_words(text, verbose)
            features = self.gen_ngram_feature(valid_tokens, duplicate)
            feature_list.append(features)
        cost_time = time.time() - start_time
        data_num = len(text_list)
        speed = data_num / cost_time
        logging.info("transform data num = {}, cost time = {:.4f}, " \
                "speed = {:.2f}/s".format(data_num, cost_time, speed))
        return feature_list

    def save(self, tokenizer_path, overwrite=False):
        """����tokenizer����
        """
        config_str = json.dumps(self.config_dict)
        if os.path.exists(tokenizer_path) and not overwrite:
            logging.error("target file exists, specified if need to overwrite")
        else:
            with open(tokenizer_path, "w") as wf:
                wf.write(config_str)
            logging.info("save config: {}".format(self.config_dict))
            logging.info("save tokenizer config into file: {}".format(tokenizer_path))

    @classmethod
    def load(cls, config_path, **kwargs):
        """��������������
        """
        if not os.path.exists(config_path):
            raise ValueError('no config file in config path: %s' % config_path)
        with open(config_path) as rf:
            config = dict(json.loads(rf.read()), **kwargs)
        logging.info("load config: {}".format(config))
        return cls(**config)

    def destroy(self):
        """wordsegger����
        """
        self._segger.destroy()


if __name__ == "__main__":
    tokenizer = LRTokenizer(
            seg_method="word_seg",
            stopword_path="./dict/stopword_shrink.txt",
            #jieba_tmp_dir="./dict/jieba_tmp",
            )

    tests = ["�����Ƿ��ܹ������д�",
             "����һ��]ͣ�ô�  ���߼�����,���ܹ������д�",
             u"��Ϸ���������,���ȶ�,����ƽ,���������������Ϸ ������ƽ̨ 24Сʱ����,\
             רҵ����,�����������Ϸ,ע�ἴ�ͺ������! ����ͬʱ����,�������ڴ����ļ���!��Ķ�Ŷ!"]

    feature_vec = tokenizer.transform(tests)

    for test, cur_feature_list in zip(tests, feature_vec):
        print("origin text: {}".format(test))
        print("cur feature list: {}".format(cur_feature_list))

    tokenizer_save_path = "./test/output/test_tokenizer"

    tokenizer.save(tokenizer_save_path, overwrite=True)

    # ���¼���
    new_tokenizer = LRTokenizer(ngram=10, feature_min_length=10)
    logging.info("config before load: {}".format(new_tokenizer.config_dict))
    LRTokenizer.load(tokenizer_save_path, ngram=5)
