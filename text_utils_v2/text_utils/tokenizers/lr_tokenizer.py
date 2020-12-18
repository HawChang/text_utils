#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   lr_tokenizer.py
Author:   zhanghao55@baidu.com
Date  :   20/09/28 14:22:12
Desc  :   
"""

import jieba
import json
import logging
import os
import sys
import time

#_cur_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append("%s/../" % _cur_dir)

from text_utils.utils.data_io import read_from_file


class LRTokenizer():
    def __init__(self,
            stopword_path=None,
            encoding="gb18030",
            lowercase=True,
            ngram=3,
            feature_min_length=2,
            jieba_tmp_dir=None,
            ):
        """
        初始化tokenizer
        [in] stopword_path: str, 停用词文件地址，为None则无停用词
             encoding: str, 停用词文件编码
             lowercase: bool, 是否统一转为小写
             n_gram: str: int, ngram参数
             feature_min_length: int, 特征最小长度
             jieba_tmp_dir: str, 结巴分词的临时文件夹地址
        """
        # 保存各配置信息
        self._stopword_path = stopword_path
        self._encoding = encoding
        self._lowercase = lowercase
        self._ngram = 3 if ngram is None or ngram < 0 else ngram
        self._feature_min_length = 0 if feature_min_length is None else feature_min_length
        self._jieba_tmp_dir = jieba_tmp_dir

        self.config_dict = {
                "stopword_path": self._stopword_path,
                "encoding": self._encoding,
                "lowercase": self._lowercase,
                "ngram": self._ngram,
                "feature_min_length": self._feature_min_length,
                "jieba_tmp_dir": self._jieba_tmp_dir,
                }

        if jieba_tmp_dir is not None:
            if os.path.isdir(jieba_tmp_dir):
                jieba.dt.tmp_dir = jieba_tmp_dir
            else:
                os.mkdir(jieba_tmp_dir)
                logging.error("creat tmp dir for jieba: {}".format(jieba_tmp_dir))

        # 加载切词工具和停用词信息
        #self._segger = WordSegger(self.seg_method, self.segdict_path)
        self._stopwords = set() if stopword_path is None \
                else set(read_from_file(stopword_path, encoding=encoding))

    def seg_words(self, text, verbose=False):
        """切词
        [in] text: str, 待切词字符串, unicode或gb18030编码
        [out] valid_tokens: list[str], 切词结果, unicode编码
        """
        # 得到切词结果 切词结果为字符串迭代器 unicode编码
        #tokens = self._segger.seg_words(text)
        tokens = jieba.cut(text)
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

    def encode(self, text, verbose=False, duplicate=False):
        """根据字符串生成ngram特征,适配ernie_tokenizer的encode函数
        [in] text: str, 字符串, unicode或gb18030编码
             verbose: bool, true则显示执行细节信息
             duplicate: bool, true则生成所有特征，不去重
        [out] feature_list: list[list[str]], 特征列表的列表,只取第一个元素即可(适配ernie_tokenizer的encode), unicode编码
        """
        if self._lowercase:
            text = text.lower()
        valid_tokens = self.seg_words(text, verbose)
        features = self.gen_ngram_feature(valid_tokens, duplicate)
        return [features]

    def transform(self, text_list, verbose=False, duplicate=False):
        """根据字符串生成ngram特征
        [in] text_list: list[str], 字符串列表, unicode或gb18030编码
             verbose: bool, true则显示执行细节信息
             duplicate: bool, true则生成所有特征，不去重
        [out] feature_list: list[list[str]], 特征列表的列表, unicode编码
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
        """保存tokenizer配置
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
        """加载特征生成类
        """
        if not os.path.exists(config_path):
            raise ValueError('no config file in config path: %s' % config_path)
        with open(config_path) as rf:
            config = dict(json.loads(rf.read()), **kwargs)
        logging.info("load config: {}".format(config))
        return cls(**config)


if __name__ == "__main__":
    tokenizer = LRTokenizer(
            stopword_path="./dict/stopword_shrink.txt",
            jieba_tmp_dir="./dict/jieba_tmp",
            )

    tests = ["测试是否能够正常切词",
             "测试一下]停用词  的逻辑，是,否？能够正常切词",
             u"游戏极多的棋牌,极稳定,极公平,人气极多的棋牌游戏 的棋牌平台 24小时服务,\
             专业贴心,处理的棋牌游戏,注册即送海量金币! 万人同时在线,的棋牌期待您的加入!你的懂哦!"]

    feature_vec = tokenizer.transform(tests)

    for test, cur_feature_list in zip(tests, feature_vec):
        print("origin text: {}".format(test))
        print("cur feature list: {}".format(cur_feature_list))

    tokenizer_save_path = "./test/output/test_tokenizer"

    tokenizer.save(tokenizer_save_path, overwrite=True)

    # 重新加载
    new_tokenizer = LRTokenizer(ngram=10, feature_min_length=10)
    logging.info("config before load: {}".format(new_tokenizer.config_dict))
    LRTokenizer.load(tokenizer_save_path, ngram=5)
