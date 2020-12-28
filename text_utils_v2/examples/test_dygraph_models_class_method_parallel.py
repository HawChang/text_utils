#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_dygraph_models.py
Author:   zhanghao55@baidu.com
Date  :   20/12/16 18:41:47
Desc  :   
"""

import os
import sys
import logging
import paddle.fluid as F
import paddle.fluid.dygraph as D
import time
import unittest

from sklearn.model_selection import train_test_split

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.models.dygraph.base_model import ClassificationModel
from text_utils.models.dygraph.base_model import model_parallelized
from text_utils.models.dygraph.nets.textcnn import TextCNNClassifier
from text_utils.models.dygraph.nets.gru import GRU
from text_utils.models.dygraph.nets.ernie_for_sequence_classification import ErnieSequenceClassificationCustomized
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.utils.data_io import get_attr_values
from text_utils.utils.data_io import write_to_file
from text_utils.utils.label_encoder import LabelEncoder
from text_utils.utils.logger import init_log

init_log()

class TestDygraphModelsParallelized(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestDygraphModelsParallelized.test_output_dir = os.path.join(test_root, "output/test_dygraph_models/")
        if not os.path.isdir(TestDygraphModelsParallelized.test_output_dir):
            os.mkdir(TestDygraphModelsParallelized.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/classification_data/toutiao_news")

        test_size = 0.15
        random_state = 1
        shuffle = True
        example_num = 5

        label_id_path = os.path.join(test_data_dir, "class_id.txt")
        TestDygraphModelsParallelized.label_encoder = LabelEncoder(label_id_path, isFile=True)
        logging.info("label num = {}".format(TestDygraphModelsParallelized.label_encoder.size()))

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        text_list, label_list = \
                get_attr_values(data_path, fetch_list=["text", "label"], encoding="utf-8")
        logging.info("data num = {}".format(len(text_list)))

        TestDygraphModelsParallelized.tokenizer = ErnieTokenizer.load("./dict/vocab.txt")
        text_ids = TestDygraphModelsParallelized.tokenizer.transform(text_list)

        label_ids = [TestDygraphModelsParallelized.label_encoder.transform(label_name) for label_name in label_list]

        TestDygraphModelsParallelized.train_text, TestDygraphModelsParallelized.test_text, \
            train_x, test_x, train_y, test_y = \
            train_test_split(text_list, text_ids, label_ids,
                             test_size=test_size, random_state=random_state, shuffle=shuffle)
        logging.info("train num = {}".format(len(train_y)))
        logging.info("test num = {}".format(len(test_y)))

        TestDygraphModelsParallelized.train_data = list(zip(train_x, train_y))
        TestDygraphModelsParallelized.eval_data = list(zip(test_x, test_y))

        place = F.CUDAPlace(D.ParallelEnv().dev_id)
        with D.guard(place):
            TestDygraphModelsParallelized.strategy = D.prepare_context()

        logging.info(u"数据样例")
        for index, (text, (token_ids, label_id)) in enumerate(zip(
                TestDygraphModelsParallelized.train_text[:example_num],
                TestDygraphModelsParallelized.train_data[:example_num],
                )):
            label_name = TestDygraphModelsParallelized.label_encoder.inverse_transform(label_id)
            logging.info("example #{}:".format(index))
            logging.info("label: {}".format(label_name))
            logging.info("text: {}".format(text.encode("utf-8")))
            logging.info("token_ids: {}".format(token_ids))

    def test_textcnn_parallelized(self):
        # 多卡运行时 embedding层的is_sparse参数需要为False 意为梯度更新时不使用稀疏更新
        # 因为多卡训练在收集梯度时不能处理稀疏梯度
        textcnn_config = {
                "num_class": TestDygraphModelsParallelized.label_encoder.size(),
                "vocab_size": TestDygraphModelsParallelized.tokenizer.size(),
                "emb_dim" : 512,
                "num_filters": 256,
                "fc_hid_dim": 512,
                "num_channels":1,
                "win_size_list": [3],
                "is_sparse": False,
                "use_cudnn": True,
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "textcnn"),
                "best_model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "textcnn_best"),
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 5e-4,
                "max_seq_len": 300,
                "print_step": 200,
                "load_best_model": True,
                }

        start_time = time.time()
        class TextCNNModel(ClassificationModel):
            @model_parallelized(TestDygraphModelsParallelized.strategy)
            def build(self, **model_config):
                self.model = TextCNNClassifier(**model_config)
                self.built = True

        place = F.CUDAPlace(D.ParallelEnv().dev_id)
        with D.guard(place):
            textcnn_model = TextCNNModel()
            textcnn_model.build(**textcnn_config)
            best_acc = textcnn_model.train(
                    TestDygraphModelsParallelized.train_data, TestDygraphModelsParallelized.eval_data,
                    label_encoder=TestDygraphModelsParallelized.label_encoder,
                    **run_config)
        logging.warning("textcnn parallelized best train score: {}, cost time: {}s".format(best_acc, time.time()- start_time))

    def test_gru_parallelized(self):
        gru_config = {
                "num_class": TestDygraphModelsParallelized.label_encoder.size(),
                "vocab_size": TestDygraphModelsParallelized.tokenizer.size(),
                "emb_dim" : 512,
                "gru_dim" : 256,
                "fc_hid_dim": 512,
                "is_sparse": False,
                "bi_direction": True,
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "gru"),
                "best_model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "gru_best"),
                "epochs": 2,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                "learning_rate": 5e-4,
                }

        start_time = time.time()
        class GRUModel(ClassificationModel):
            @model_parallelized(TestDygraphModelsParallelized.strategy)
            def build(self, **model_config):
                self.model = GRU(**model_config)
                self.built = True

        place = F.CUDAPlace(D.ParallelEnv().dev_id)
        with D.guard(place):
            gru_model = GRUModel()
            gru_model.build(**gru_config)
            best_acc = gru_model.train(
                    TestDygraphModelsParallelized.train_data, TestDygraphModelsParallelized.eval_data,
                    label_encoder=TestDygraphModelsParallelized.label_encoder,
                    **run_config)
        logging.warning("gru parallelized best train score: {}, cost time: {}s".format(best_acc, time.time()- start_time))

    def test_ernie_parallelized(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                "num_labels": TestDygraphModelsParallelized.label_encoder.size(),
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "ernie"),
                "best_model_save_path": os.path.join(TestDygraphModelsParallelized.test_output_dir, "ernie_best"),
                "epochs": 2,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": False,
                }

        start_time = time.time()
        class ErnieClassificationModel(ClassificationModel):
            @model_parallelized(TestDygraphModelsParallelized.strategy)
            def build(self, **model_config):
                self.model = ErnieSequenceClassificationCustomized.from_pretrained(**model_config)
                self.built = True

        place = F.CUDAPlace(D.ParallelEnv().dev_id)
        with D.guard(place):
            ernie_classification_model = ErnieClassificationModel()
            ernie_classification_model.build(**ernie_config)
            best_acc = ernie_classification_model.train(
                    TestDygraphModelsParallelized.train_data, TestDygraphModelsParallelized.eval_data,
                    label_encoder=TestDygraphModelsParallelized.label_encoder,
                    **run_config)
        logging.warning("ernie parallelized best train score: {}, cost time: {}s".format(best_acc, time.time()- start_time))


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestDygraphModelsParallelized("test_textcnn_parallelized"))
    #suit.addTest(TestDygraphModelsParallelized("test_gru_parallelized"))
    #suit.addTest(TestDygraphModelsParallelized("test_ernie_parallelized"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

