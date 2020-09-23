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

from utils.data_io import get_data
from feature.feature_generator import FeatureGenerator
from model.word_vec.fasttext import fasttext_training
from model.word_vec.fasttext import load_fasttext_model
from utils.logger import init_log

init_log()


fastttext_data_dir = "test/data/jinyong"
fasttext_model_path = "test/output/fasttext_model"

def get_similar(text, model):
    if type(text) != type(u""):
        text = text.decode("gb18030", "ignore")
    print("text: %s" % text)
    res_list = model.wv.most_similar(text)
    print(u"原词：%s " % text)
    print(u"相似词\t相似度")
    for key, value in res_list:
        print("%s\t%f" % (key, value))


class TestFasttext(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fastext_train(self):
        generator = FeatureGenerator(
                seg_method="word_seg",
                segdict_path="dict/chinese_gbk",
                stopword_path="dict/stopword_shrink.txt",
                encoding="gb18030",
                )

        def process_func(line):
            if len(line) < 6:
                return None
            seg_list = generator.seg_words(line)
            return [x.strip() for x in seg_list if len(x.strip()) > 0]

        # 传入的data_iter 一定是要可调用的 因为会用两次
        def DataIter():
            data_iter = get_data(
                    data_path=fastttext_data_dir,
                    read_func=process_func,
                    encoding="gb18030",
                    )
            return data_iter

        data_iter = get_data(
                data_path=fastttext_data_dir,
                read_func=process_func,
                encoding="gb18030",
                )

        logging.info("fasttext model training tsart")
        start_time = time.time()
        fast_text_model = fasttext_training(
                DataIter,
                model_save_path=fasttext_model_path,
                emb_size=128,
                epochs=5)
        logging.info("cost time = %.4fs" % (time.time() - start_time))

    def test_load_fasttext(self):
        ft_model = load_fasttext_model(fasttext_model_path)

        get_similar(u"郭靖", ft_model)
        get_similar(u"杨过", ft_model)
        get_similar(u"南帝", ft_model)
        get_similar(u"萧峰", ft_model)
        get_similar(u"乔峰", ft_model)

if __name__ == "__main__":
    # 运行所有测试用例
    unittest.main()

    # 运行指定测试用例
    # 构造测试集
    #suit = unittest.TestSuite()
    #suit.addTest(TestFasttext("test_load_fasttext"))
    #runner = unittest.TextTestRunner()
    #runner.run(suit)

