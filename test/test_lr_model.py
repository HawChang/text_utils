#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_lr_model.py
Author:   zhanghao55@baidu.com
Date  :   20/09/28 11:32:25
Desc  :   
"""

import codecs
import logging
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from sklearn.model_selection import train_test_split
import time
import unittest

#from utils.data_io import get_data
from utils.data_io import get_attr_values
from utils.label_encoder import LabelEncoder
from utils.logger import init_log

from model.lr_tokenizer import LRTokenizer
from model.lr_model import LRModel

init_log()


class TestLRModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = "./test/data/classification/data.txt"
        test_size = 0.2
        random_state = 1
        shuffle = True
        example_num = 5

        tokenizer_path = "./test/output/lr_tokenizer_config"
        label_id_path = "./test/data/classification/class_id.txt"
        TestLRModel.label_encoder = LabelEncoder(label_id_path, isFile=True)
        logging.info("label num = {}".format(TestLRModel.label_encoder.size()))

        # 加载数据
        label_list, text_list = get_attr_values(data_path, fetch_list=["label", "text"], encoding="utf-8")
        logging.info("data num = {}".format(len(label_list)))

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
                train_test_split(text_list, feature_list, label_ids, \
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

        logging.info("in test_lr_model_train")
        lr_model = LRModel()
        #train_str_list = [" ".join(x) for x in self.train_x]
        lr_model.train(self.train_x, self.train_y)

        model_path = "./test/output/lr_feature_weight.model"
        lr_model.save(model_path)

        pred_res = lr_model.predict(self.test_x)
        res_before_load = [[res[0][0], res[0][1]] if len(res) != 0 else ["0", "NULL"] for res in pred_res]

        lr_model = LRModel()
        lr_model.load(model_path)
        pred_res = lr_model.predict(self.test_x)
        res_after_load = [[res[0][0], res[0][1]] if len(res) != 0 else ["0", "NULL"] for res in pred_res]

        eval_res_path = "./test/output/eval_res.txt"
        with codecs.open(eval_res_path, "w", "gb18030") as wf:
            for (res_before_label, res_before_rate), (res_after_label, res_after_rate), label_id, text \
                    in zip(res_before_load, res_after_load, self.test_y, self.test_text):
                res_before_label_name = self.label_encoder.inverse_transform(int(res_before_label))
                res_after_label_name = self.label_encoder.inverse_transform(int(res_after_label))
                label_name = self.label_encoder.inverse_transform(label_id)
                wf.write("\t".join([
                        res_before_label_name,
                        res_before_rate,
                        res_after_label_name,
                        res_after_rate,
                        label_name,
                        text
                        ]) + "\n")

        #eval_res_path = "./test/output/eval_res.txt"
        #eval_diff_path = "./test/output/eval_diff.txt"
        #acc = lr_model.eval(self.test_x, self.test_y,
        #        pred_res_path=eval_res_path,
        #        pred_diff_path=eval_diff_path)

        #pred_res = lr_model.pred(self.test_x)

if __name__ == "__main__":
    # 运行所有测试用例
    unittest.main()

    # 运行指定测试用例
    # 构造测试集
    #suit = unittest.TestSuite()
    #suit.addTest(TestLRModel("test_load_fasttext"))
    #runner = unittest.TextTestRunner()
    #runner.run(suit)

