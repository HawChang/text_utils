#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_ernie_feature_extract.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 11:15:48
Desc  :   
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import logging
import paddle.fluid.dygraph as D
#import time
import unittest

from ernie.modeling_ernie import ErnieModel

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.utils.data_io import get_attr_values
#from text_utils.utils.data_io import write_to_file
from text_utils.models.dygraph.train_infer_utils import batch_infer
from text_utils.utils.logger import init_log

init_log()

class TestErnieExtractFeature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestErnieExtractFeature.test_output_dir = os.path.join(test_root, "output/test_dygraph_models/")
        if not os.path.isdir(TestErnieExtractFeature.test_output_dir):
            os.mkdir(TestErnieExtractFeature.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/classification_data/toutiao_news")

        example_num = 5

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        TestErnieExtractFeature.text_list, TestErnieExtractFeature.keywords_list = \
                get_attr_values(data_path, fetch_list=["text", "keywords"], encoding="utf-8")
        logging.info("data num = {}".format(len(TestErnieExtractFeature.text_list)))

        TestErnieExtractFeature.text_list = TestErnieExtractFeature.text_list[:100]
        TestErnieExtractFeature.keywords_list = TestErnieExtractFeature.keywords_list[:100]

        TestErnieExtractFeature.tokenizer = ErnieTokenizer.load("./dict/vocab.txt")
        TestErnieExtractFeature.text_ids = TestErnieExtractFeature.tokenizer.transform(TestErnieExtractFeature.text_list)
        #logging.info("text_ids: {}".format(TestErnieExtractFeature.text_ids))

        logging.info(u"数据样例")
        for index, (text, token_ids) in enumerate(zip(
                TestErnieExtractFeature.text_list[:example_num],
                TestErnieExtractFeature.text_ids[:example_num],
                )):
            logging.info("example #{}:".format(index))
            logging.info("text: {}".format(text.encode("utf-8")))
            logging.info("token_ids: {}".format(token_ids))

    def test_ernie_extract_feature(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                }

        with D.guard():
            ernie = ErnieModel.from_pretrained(**ernie_config)

            res = batch_infer(ernie, TestErnieExtractFeature.text_ids, batch_size=3, with_label=False, logits_softmax=None)
            logging.info("len res: {}".format(len(res)))

            for (pooled_encode_vec, sequence_encode_vec), text in zip(res, TestErnieExtractFeature.text_list):
                logging.info("text: {}".format(text.encode("utf-8")))
                logging.info("pooled_encode_vec shape: {}".format(pooled_encode_vec.shape))
                logging.info("sequence_encode_vec shape: {}".format(sequence_encode_vec.shape))


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestErnieExtractFeature("test_ernie_extract_feature"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

