#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/9/29 4:11 PM
# @Author : ZhangHao
# @File   : test_textcnn.py
# @Desc   : 


import logging
import os
import paddle.fluid as F
import paddle.fluid.dygraph as D
import sys
import unittest
from sklearn.model_selection import train_test_split

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

from utils.data_io import get_attr_values
from utils.label_encoder import LabelEncoder
from utils.logger import init_log
from utils.dygraph import train
from utils.ernie_tokenizer import ErnieTokenizer
from model.dygraph.textcnn import TextCNN
init_log()


class TestTextCNNModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_data_dir = "./test/data/classification_data/toutiao_news"
        TestTextCNNModel.test_output_dir = "./test/output"
        pretrain_dir = "./test/pretrain"

        test_size = 0.2
        random_state = 1
        shuffle = True
        example_num = 50

        label_id_path = os.path.join(test_data_dir, "class_id.txt")
        TestTextCNNModel.label_encoder = LabelEncoder(label_id_path, isFile=True)
        logging.info("label num = {}".format(TestTextCNNModel.label_encoder.size()))

        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        label_list, text_list = get_attr_values(data_path, fetch_list=["label", "text"], encoding="utf-8")
        logging.info("data num = {}".format(len(label_list)))

        vocab_path = os.path.join(pretrain_dir, "ernie-1.0/vocab.txt")
        tokenizer = ErnieTokenizer.from_pretrained(vocab_path)
        id_2_token = {v: k for k, v in tokenizer.vocab.items()}
        text_id_list = [tokenizer.encode(x)[0] for x in text_list]

        label_ids = [TestTextCNNModel.label_encoder.transform(label_name) for label_name in label_list]

        TestTextCNNModel.train_text, TestTextCNNModel.test_text, TestTextCNNModel.train_x, \
            TestTextCNNModel.test_x, TestTextCNNModel.train_y, TestTextCNNModel.test_y = \
            train_test_split(text_list, text_id_list, label_ids,
                             test_size=test_size, random_state=random_state, shuffle=shuffle)
        logging.info("train num = {}".format(len(TestTextCNNModel.train_y)))
        logging.info("test num = {}".format(len(TestTextCNNModel.test_y)))

        logging.info("数据样例")
        for index, (label_id, text, text_ids) in enumerate(zip(
                TestTextCNNModel.train_y[:example_num],
                TestTextCNNModel.train_text[:example_num],
                TestTextCNNModel.train_x[:example_num],
                )):
            label_name = TestTextCNNModel.label_encoder.inverse_transform(label_id)
            tokens = "/ ".join([id_2_token[x] for x in text_ids])
            logging.info("example #{}:".format(index))
            logging.info("label: {}".format(label_name))
            logging.info("text: {}".format(text))
            logging.info("feature: {}".format(tokens))

    def test_textcnn_train(self):
        vocab_size = 18000
        emb_dim = 512
        num_filters = 256
        fc_hid_dim = 512
        use_cudnn = True
        learning_rate = 5e-4
        
        model_path = os.path.join(self.test_output_dir, "textcnn/textcnn_model")
        best_model_path = os.path.join(self.test_output_dir, "textcnn/textcnn_model_best")
        with D.guard():
            text_cnn = TextCNN(
                num_class=self.label_encoder.size(),
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                num_filters=num_filters,
                fc_hid_dim=fc_hid_dim,
                use_cudnn=use_cudnn)

            optimizer = F.optimizer.Adam(
                learning_rate=learning_rate,
                parameter_list=text_cnn.parameters())

            train_data = list(zip(self.train_x, self.train_y))
            eval_data = list(zip(self.test_x, self.test_y))

            textcnn_best_acc = train(text_cnn, optimizer, train_data, eval_data, self.label_encoder,
                                     model_save_path=model_path, best_model_save_path= best_model_path,
                                     best_acc=0, epochs=5, batch_size=32, max_seq_len=300)

            logging.info("textcnn best score: {}".format(textcnn_best_acc))


if __name__ == "__main__":
    # 运行所有测试用例
    unittest.main()

    # 运行指定测试用例
    # 构造测试集
    #suit = unittest.TestSuite()
    #suit.addTest(TestLRModel("test_load_fasttext"))
    #runner = unittest.TextTestRunner()
    #runner.run(suit)