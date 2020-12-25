#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_dygraph_models.py
Author:   zhanghao55@baidu.com
Date  :   20/12/16 18:41:47
Desc  :   
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import logging
import paddle.fluid as F
import paddle.fluid.dygraph as D
#import time
import unittest

from sklearn.model_selection import train_test_split

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.utils.data_io import get_attr_values
from text_utils.utils.data_io import write_to_file
from text_utils.utils.label_encoder import LabelEncoder
from text_utils.utils.logger import init_log
from text_utils.models.dygraph.base_model_parallel import ClassificationModel
from text_utils.models.dygraph.base_model_parallel import model_parrallel
from text_utils.models.dygraph.nets.textcnn import TextCNNClassifier
from text_utils.models.dygraph.nets.gru import GRU
from text_utils.models.dygraph.nets.ernie_for_sequence_classification import ErnieSequenceClassificationCustomized
from text_utils.models.dygraph.train_infer_utils import train
from text_utils.utils.data_io import load_model

init_log()

class TestDygraphModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestDygraphModels.test_output_dir = os.path.join(test_root, "output/test_dygraph_models/")
        if not os.path.isdir(TestDygraphModels.test_output_dir):
            os.mkdir(TestDygraphModels.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/classification_data/toutiao_news")

        test_size = 0.15
        random_state = 1
        shuffle = True
        example_num = 5

        label_id_path = os.path.join(test_data_dir, "class_id.txt")
        TestDygraphModels.label_encoder = LabelEncoder(label_id_path, isFile=True)
        logging.info("label num = {}".format(TestDygraphModels.label_encoder.size()))

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        text_list, label_list = \
                get_attr_values(data_path, fetch_list=["text", "label"], encoding="utf-8")
        logging.info("data num = {}".format(len(text_list)))

        TestDygraphModels.tokenizer = ErnieTokenizer.load("./dict/vocab.txt")
        text_ids = TestDygraphModels.tokenizer.transform(text_list)

        label_ids = [TestDygraphModels.label_encoder.transform(label_name) for label_name in label_list]

        TestDygraphModels.train_text, TestDygraphModels.test_text, \
            train_x, test_x, train_y, test_y = \
            train_test_split(text_list, text_ids, label_ids,
                             test_size=test_size, random_state=random_state, shuffle=shuffle)
        logging.info("train num = {}".format(len(train_y)))
        logging.info("test num = {}".format(len(test_y)))

        TestDygraphModels.train_data = list(zip(train_x, train_y))
        TestDygraphModels.eval_data = list(zip(test_x, test_y))

        logging.info(u"数据样例")
        for index, (text, (token_ids, label_id)) in enumerate(zip(
                TestDygraphModels.train_text[:example_num],
                TestDygraphModels.train_data[:example_num],
                )):
            label_name = TestDygraphModels.label_encoder.inverse_transform(label_id)
            logging.info("example #{}:".format(index))
            logging.info("label: {}".format(label_name))
            logging.info("text: {}".format(text.encode("utf-8")))
            logging.info("token_ids: {}".format(token_ids))

    def model_train_infer(self, model, run_config):
        load_model(model, run_config["best_model_save_path"])
        optimizer = F.optimizer.Adam(
                learning_rate=run_config["learning_rate"],
                parameter_list=model.parameters())

        best_acc = train(model, optimizer,
                TestDygraphModels.train_data, TestDygraphModels.eval_data,
                TestDygraphModels.label_encoder, best_acc=0,
                model_save_path=run_config["model_save_path"],
                best_model_save_path=run_config["best_model_save_path"],
                epochs=run_config["epochs"],
                batch_size=run_config["batch_size"],
                max_seq_len=run_config["max_seq_len"],
                print_step=run_config["print_step"],
                )
        return best_acc

    def test_textcnn(self):
        textcnn_config = {
                "num_class": TestDygraphModels.label_encoder.size(),
                "vocab_size": TestDygraphModels.tokenizer.size(),
                "emb_dim" : 512,
                "num_filters": 256,
                "fc_hid_dim": 512,
                "num_channels":1,
                "win_size_list": [3],
                "is_sparse": True,
                "use_cudnn": True,
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModels.test_output_dir, "textcnn"),
                "best_model_save_path": os.path.join(TestDygraphModels.test_output_dir, "textcnn_best"),
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 5e-4,
                "max_seq_len": 300,
                "print_step": 200,
                "load_best_model": True,
                }

        class TextCNNModel(ClassificationModel):
            @model_parrallel
            def build(self, **model_config):
                self.model = TextCNNClassifier(**model_config)
                self.built = True

        place = F.CUDAPlace(D.ParallelEnv().dev_id)
        with D.guard(place):
            textcnn_model = TextCNNModel()
            textcnn_model.build(**textcnn_config)
            best_acc = textcnn_model.train(
                    TestDygraphModels.train_data, TestDygraphModels.eval_data,
                    label_encoder=TestDygraphModels.label_encoder,
                    **run_config)
        logging.info("textcnn best train score: {}".format(best_acc))

    def test_gru(self):
        gru_config = {
                "num_class": TestDygraphModels.label_encoder.size(),
                "vocab_size": TestDygraphModels.tokenizer.size(),
                "emb_dim" : 512,
                "gru_dim" : 256,
                "fc_hid_dim": 512,
                "is_sparse": True,
                "bi_direction": True,
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModels.test_output_dir, "gru"),
                "best_model_save_path": os.path.join(TestDygraphModels.test_output_dir, "gru_best"),
                "epochs": 2,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                "learning_rate": 5e-4,
                }

        class GRUModel(ClassificationModel):
            def build(self, **model_config):
                self.model = GRU(**model_config)
                self.built = True

        with D.guard():
            gru_model = GRUModel()
            gru_model.build(**gru_config)
            best_acc = gru_model.train(
                    TestDygraphModels.train_data, TestDygraphModels.eval_data,
                    label_encoder=TestDygraphModels.label_encoder,
                    **run_config)
        logging.info("gru best train score: {}".format(best_acc))

    def test_ernie(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                "num_labels": TestDygraphModels.label_encoder.size(),
                }

        run_config = {
                "model_save_path": os.path.join(TestDygraphModels.test_output_dir, "ernie"),
                "best_model_save_path": os.path.join(TestDygraphModels.test_output_dir, "ernie_best"),
                "epochs": 2,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 100,
                "learning_rate": 5e-5,
                }

        class ErnieClassificationModel(ClassificationModel):
            def build(self, **model_config):
                self.model = ErnieSequenceClassificationCustomized.from_pretrained(**model_config)
                self.built = True

        with D.guard():
            ernie_classification_model = ErnieClassificationModel()
            ernie_classification_model.build(**ernie_config)
            best_acc = ernie_classification_model.train(
                    TestDygraphModels.train_data, TestDygraphModels.eval_data,
                    label_encoder=TestDygraphModels.label_encoder,
                    **run_config)
        logging.info("ernie best train score: {}".format(best_acc))


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestDygraphModels("test_textcnn"))
    #suit.addTest(TestDygraphModels("test_gru"))
    #suit.addTest(TestDygraphModels("test_ernie"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

