#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_lr_model.py
Author:   zhanghao55@baidu.com
Date  :   20/09/28 11:32:25
Desc  :   
"""

import logging
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from sklearn.model_selection import train_test_split
import unittest

from utils.data_io import get_attr_values
from utils.label_encoder import LabelEncoder
from utils.logger import init_log

from model.lr_tokenizer import LRTokenizer
from model.lr_model import LRModel

init_log()


class TestLRModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_data_dir = "./test/data/classification_data/toutiao_news"
        TestLRModel.test_output_dir = "./test/output"

        test_size = 0.2
        random_state = 1
        shuffle = True
        example_num = 5

        label_id_path = os.path.join(test_data_dir, "class_id.txt")
        TestLRModel.label_encoder = LabelEncoder(label_id_path, isFile=True)
        logging.info("label num = {}".format(TestLRModel.label_encoder.size()))

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        label_list, text_list = get_attr_values(data_path, fetch_list=["label", "text"], encoding="utf-8")
        logging.info("data num = {}".format(len(label_list)))

        tokenizer_path = os.path.join(TestLRModel.test_output_dir, "lr_tokenizer.config")
        if os.path.exists(tokenizer_path):
            TestLRModel.tokenizer = LRTokenizer.load(tokenizer_path)
        else:
            TestLRModel.tokenizer = LRTokenizer(
                    stopword_path="./dict/stopword_shrink.txt",
                    jieba_tmp_dir="./dict/jieba_tmp",
                    )
            TestLRModel.tokenizer.save(tokenizer_path)
        feature_list = TestLRModel.tokenizer.transform(text_list)

        label_ids = [TestLRModel.label_encoder.transform(label_name) for label_name in label_list]

        TestLRModel.train_text, TestLRModel.test_text, TestLRModel.train_x, \
            TestLRModel.test_x, TestLRModel.train_y, TestLRModel.test_y = \
            train_test_split(text_list, feature_list, label_ids,
                             test_size=test_size, random_state=random_state, shuffle=shuffle)
        logging.info("train num = {}".format(len(TestLRModel.train_y)))
        logging.info("test num = {}".format(len(TestLRModel.test_y)))

        logging.info("数据样例")
        for index, (label_id, text, feature) in enumerate(zip(
                TestLRModel.train_y[:example_num],
                TestLRModel.train_text[:example_num],
                TestLRModel.train_x[:example_num],
                )):
            label_name = TestLRModel.label_encoder.inverse_transform(label_id)
            logging.info("example #{}:".format(index))
            logging.info("label: {}".format(label_name))
            logging.info("text: {}".format(text))
            logging.info("feature: {}".format(feature))

    def test_lr_model_train(self):
        self.model_path = os.path.join(self.test_output_dir, "lr_feature_weight.model")
        lr_model = LRModel()
        lr_model.train(self.train_x, self.train_y)

        lr_model.save(self.model_path)

    def test_lr_model_eval(self):
        eval_res_path = os.path.join(self.test_output_dir, "eval_res.txt")
        eval_diff_path = os.path.join(self.test_output_dir, "eval_diff.txt")
        lr_model = LRModel.load(model_path=self.model_path)
        lr_model.eval(
                eval_feature=self.test_x,
                eval_label=[self.label_encoder.inverse_transform(x) for x in self.test_y],
                eval_text_info=self.test_text,
                default_label=self.label_encoder.other_label_id,
                label_trans_func=lambda x: self.label_encoder.inverse_transform(int(x)),
                eval_res_path=eval_res_path,
                eval_diff_path=eval_diff_path,
                )


if __name__ == "__main__":
    # 运行所有测试用例
    unittest.main()

    # 运行指定测试用例
    # 构造测试集
    #suit = unittest.TestSuite()
    #suit.addTest(TestLRModel("test_load_fasttext"))
    #runner = unittest.TextTestRunner()
    #runner.run(suit)

