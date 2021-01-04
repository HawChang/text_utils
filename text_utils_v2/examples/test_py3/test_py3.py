#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_py3.py
Author:   zhanghao55@baidu.com
Date  :   21/01/04 17:11:20
Desc  :   
"""

import logging
import numpy as np
import paddle.fluid.dygraph as D
import time
import unittest

from base_model import ClassificationModel
from ernie_tokenizer import ErnieTokenizer
from gru import GRUClassifier
from lstm import DynamicLSTMClassifier


def train_lstm(num_class, vocab_size, data):
    logging.warning("lstm train start")
    lstm_config = {
            "num_class": num_class,
            "vocab_size": vocab_size,
            "emb_dim" : 512,
            "lstm_dim" : 256,
            "fc_hid_dim": 512,
            "is_sparse": True,
            "bi_direction": True,
            }

    run_config = {
            "epochs": 2,
            "batch_size": 32,
            "max_seq_len": 100,
            "print_step": 1,
            "learning_rate": 5e-4,
            "load_best_model": False,
            }

    start_time = time.time()
    class LSTMModel(ClassificationModel):
        def build(self, **model_config):
            self.model = DynamicLSTMClassifier(**model_config)
            self.built = True

    with D.guard():
        lstm_model = LSTMModel()
        lstm_model.build(**lstm_config)
        best_acc = lstm_model.train(
                data, data,
                label_encoder=None,
                **run_config)
    logging.warning("lstm best train score: {}, cost time: {}s".format(best_acc, time.time()- start_time))


def train_gru(num_class, vocab_size, data):
    logging.warning("gru train start")
    gru_config = {
            "num_class": num_class,
            "vocab_size": vocab_size,
            "emb_dim" : 512,
            "gru_dim" : 256,
            "fc_hid_dim": 512,
            "is_sparse": True,
            "bi_direction": True,
            }

    run_config = {
            "epochs": 2,
            "batch_size": 32,
            "max_seq_len": 100,
            "print_step": 1,
            "learning_rate": 5e-4,
            "load_best_model": False,
            }

    start_time = time.time()
    class GRUModel(ClassificationModel):
        def build(self, **model_config):
            self.model = GRUClassifier(**model_config)
            self.built = True

    with D.guard():
        gru_model = GRUModel()
        gru_model.build(**gru_config)
        best_acc = gru_model.train(
                data, data,
                label_encoder=None,
                **run_config)
    logging.warning("gru best train score: {}, cost time: {}s".format(best_acc, time.time()- start_time))


class TestDygraphModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text_list = [
                "为村长生个儿子（现代故事）",
                "东北一家人在杭居住十一年：还是吃不惯酱鸭",
                "哪些是你后悔知道的太晚的书法窍门？",
                "吹箫能引凤，快婿长乘龙，下句是什么？",
                ]

        label_list = [
                "news_story",
                "news_story",
                "news_culture",
                "news_culture",
                ]

        for _ in range(8):
            text_list.extend(text_list)
            label_list.extend(label_list)

        logging.warning("text_list size: {}".format(len(text_list)))

        TestDygraphModels.label_dict = {
                "news_story": 0,
                "news_culture": 1,
                }

        TestDygraphModels.tokenizer = ErnieTokenizer.load("../dict/vocab.txt")

        text_ids = TestDygraphModels.tokenizer.transform(text_list)
        label_ids = [TestDygraphModels.label_dict[x] for x in label_list]
        TestDygraphModels.data = list(zip(text_ids, label_ids))

        for index, (text, (text_ids, label_id)) in enumerate(zip(text_list, TestDygraphModels.data)):
            if index > 5:
                break
            logging.warning("example #{}:".format(index))
            logging.warning("label: {}".format(label_id))
            logging.warning("text: {}".format(text))
            logging.warning("text_ids: {}".format(text_ids))

    def test_lstm(self):
        train_lstm(len(TestDygraphModels.label_dict), TestDygraphModels.tokenizer.size(), TestDygraphModels.data)

    def test_gru(self):
        train_gru(len(TestDygraphModels.label_dict), TestDygraphModels.tokenizer.size(), TestDygraphModels.data)


if __name__ == "__main__":
    suit = unittest.TestSuite()
    suit.addTest(TestDygraphModels("test_lstm"))
    suit.addTest(TestDygraphModels("test_gru"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

