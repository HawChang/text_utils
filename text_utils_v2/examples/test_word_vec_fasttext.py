#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_word_vec_fasttext.py
Author:   zhanghao55@baidu.com
Date  :   20/07/25 16:13:36
Desc  :   
"""

import logging
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
import time
import unittest

from text_utils.utils.data_io import get_data
from text_utils.tokenizers.lr_tokenizer import LRTokenizer
from text_utils.models.word_vec.fasttext import fasttext_training
from text_utils.models.word_vec.fasttext import load_fasttext_model
from text_utils.utils.logger import init_log
from text_utils.utils.data_io import write_to_file

init_log()


fastttext_data_dir = "test/data/jinyong"
fasttext_model_path = "test/output/fasttext_model"

def get_similar(text, model):
    if type(text) != type(u""):
        text = text.decode("gb18030", "ignore")
    logging.info(u"text: %s" % text)
    res_list = model.wv.most_similar(text)
    logging.info(u"原词：%s " % text)
    logging.info(u"相似词\t相似度")
    for key, value in res_list:
        logging.info("%s\t%f" % (key, value))


class TestFasttext(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestFasttext.test_output_dir = os.path.join(test_root, "output/test_word_vec_fasttext")
        if not os.path.isdir(TestFasttext.test_output_dir):
            os.mkdir(TestFasttext.test_output_dir)

        TestFasttext.test_data_dir = os.path.join(test_root, "dataset/jinyong/")
        TestFasttext.model_path = os.path.join(TestFasttext.test_output_dir, "fasttext.model")

        tokenizer_path = os.path.join(TestFasttext.test_output_dir, "fasttext_tokenizer.config")
        if os.path.exists(tokenizer_path):
            TestFasttext.tokenizer = LRTokenizer.load(tokenizer_path)
        else:
            TestFasttext.tokenizer = LRTokenizer(
                    seg_method="jieba",
                    stopword_path="./dict/stopword_shrink.txt",
                    jieba_tmp_dir="./dict/jieba_tmp",
                    )
            TestFasttext.tokenizer.save(tokenizer_path)

        TestFasttext.seg_line = 0

        def process_func(line):
            if len(line) < 6:
                return None
            seg_list = TestFasttext.tokenizer.seg_words(line)
            if TestFasttext.seg_line % 2000 == 0:
                logging.info("text: {}".format(line.encode("utf-8")))
                logging.info("=" * 100)
                logging.info("seg list: {}".format("/ ".join(seg_list).encode("utf-8")))
            TestFasttext.seg_line += 1
            return [x.strip() for x in seg_list if len(x.strip()) > 0]

        TestFasttext.data_list = list(get_data(
            data_path=TestFasttext.test_data_dir,
            read_func=process_func,
            encoding="gb18030",
            ))

        write_to_file(TestFasttext.data_list, TestFasttext.test_output_dir + "/seg_text.txt", write_func=lambda x:" ".join(x))

    @classmethod
    def tearDownClass(cls):
        TestFasttext.tokenizer.destroy()

    def data_reader(self):
        return iter(TestFasttext.data_list)

    def test_fastext_train(self):
        logging.info("fasttext model training start")
        start_time = time.time()
        fast_text_model = fasttext_training(
                self.data_reader,
                model_save_path=TestFasttext.model_path,
                emb_size=128,
                epochs=20)
        logging.info("cost time = %.4fs" % (time.time() - start_time))

    def test_load_fasttext(self):
        ft_model = load_fasttext_model(TestFasttext.model_path)

        get_similar(u"郭靖", ft_model)
        get_similar(u"杨过", ft_model)
        get_similar(u"南帝", ft_model)
        get_similar(u"萧峰", ft_model)
        get_similar(u"乔峰", ft_model)

if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestFasttext("test_fastext_train"))
    suit.addTest(TestFasttext("test_load_fasttext"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

