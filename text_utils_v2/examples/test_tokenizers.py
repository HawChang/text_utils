#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_tokenizers.py
Author:   zhanghao55@baidu.com
Date  :   20/12/16 15:30:37
Desc  :   
"""

import logging
import os
import sys
import time
import unittest

from tqdm import tqdm

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.tokenizers.ernie_tokenizer_old import ErnieTokenizerOld
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.tokenizers.lr_tokenizer import LRTokenizer
from text_utils.utils.data_io import get_attr_values
from text_utils.utils.data_io import write_to_file
from text_utils.utils.logger import init_log

init_log()

class TestTokenizers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestTokenizers.test_output_dir = os.path.join(test_root, "output/test_tokenizers/")
        if not os.path.isdir(TestTokenizers.test_output_dir):
            os.mkdir(TestTokenizers.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/classification_data/toutiao_news")
        example_num = 5

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news.txt")
        TestTokenizers.text_list, TestTokenizers.keywords_list = \
                get_attr_values(data_path, fetch_list=["text", "keywords"], encoding="utf-8")
        logging.info("text num = {}".format(len(TestTokenizers.text_list)))

        logging.info("数据样例")
        for index, text in enumerate(TestTokenizers.text_list[:example_num]):
            logging.info("example #{}:".format(index))
            logging.info("text: {}".format(text))

    def tokenize_data(self, tokenizer, output_path):
        def tokenize_iter():
            for index, text in enumerate(tqdm(TestTokenizers.text_list)):
                yield tokenizer.encode(text)

        start_time = time.time()
        write_to_file(tokenize_iter(), output_path, write_func=lambda x: str(x), encoding="gb18030")
        logging.info("tokenize data cost time: {}s".format(time.time() - start_time))

    def tokenize_pair_data(self, tokenizer, output_path):
        def tokenize_iter():
            for text, keywords in tqdm(
                    zip(TestTokenizers.text_list, TestTokenizers.keywords_list),
                    total=len(TestTokenizers.text_list)):
                yield tokenizer.encode(text, pair=keywords, truncate_to=30)

        start_time = time.time()
        write_to_file(tokenize_iter(), output_path, write_func=lambda x: str(x), encoding="gb18030")
        logging.info("tokenize data cost time: {}s".format(time.time() - start_time))

    def test_lr_tokenizer(self):
        # 测试lr_tokenizer初始化的效果
        cur_tokenizer = LRTokenizer(
                stopword_path="./dict/stopword_shrink.txt",
                jieba_tmp_dir="./dict/jieba_tmp",
                )
        lr_tokenize_res_1_path = os.path.join(TestTokenizers.test_output_dir, "lr_tokenize_res_1.txt")
        self.tokenize_data(cur_tokenizer, lr_tokenize_res_1_path)

        tokenizer_path = os.path.join(TestTokenizers.test_output_dir, "lr_tokenizer.config")
        cur_tokenizer.save(tokenizer_path, True)

        # 测试lr_tokenizer从文件加载的效果
        cur_tokenizer = LRTokenizer.load(tokenizer_path)
        lr_tokenize_res_2_path = os.path.join(TestTokenizers.test_output_dir, "lr_tokenize_res_2.txt")
        self.tokenize_data(cur_tokenizer, lr_tokenize_res_2_path)

    def test_ernie_tokenizer(self):
        cur_tokenizer = ErnieTokenizer.load("./dict/vocab.txt")
        tokenize_res_1_path = os.path.join(TestTokenizers.test_output_dir, "ernie_tokenize_res_1.txt")
        self.tokenize_pair_data(cur_tokenizer, tokenize_res_1_path)

        cur_tokenizer_old = ErnieTokenizerOld.from_pretrained("./dict/vocab.txt")
        tokenize_res_2_path = os.path.join(TestTokenizers.test_output_dir, "ernie_tokenize_res_2.txt")
        self.tokenize_pair_data(cur_tokenizer_old, tokenize_res_2_path)


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestTokenizers("test_lr_tokenizer"))
    suit.addTest(TestTokenizers("test_ernie_tokenizer"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

