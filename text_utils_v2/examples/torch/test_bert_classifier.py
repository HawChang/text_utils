#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_bert_classifier.py
Author:   zhanghao55@baidu.com
Date  :   21/01/09 14:45:56
Desc  :   
"""

import os
import sys
import logging
import numpy as np
import json
import time
import torch
import unittest
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)
from text_utils.utils.data_io import get_attr_values
from text_utils.utils.label_encoder import LabelEncoder
from text_utils.utils.logger import init_log
from text_utils.models.torch.base_model import ClassificationModel
from text_utils.models.torch.nets.bert import BertForClassification
from text_utils.tokenizers.bert_tokenizer import BertTokenizer

init_log(stream_level=logging.INFO)


class ClassificationDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data_list) :
        ## 一般init函数是加载所有数据
        super(ClassificationDataset, self).__init__()
        # dataloader中的数据用numpy保存
        # 相关issue: https://github.com/pytorch/pytorch/issues/13246
        self.data_list = np.array(data_list)

    def __getitem__(self, index):
        # 得到单个数据
        token_ids, token_type_ids, label_id = self.data_list[index]
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": label_id
        }
        return output

    def __len__(self):
        return len(self.data_list)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return token_ids_padded, token_type_ids_padded, target_ids


def create_train_test_dataset(data_dir, tokenizer, label_encoder,
        test_size=0.15, shuffle=True, random_state=1):
    # 加载数据
    text_list, label_list = \
            get_attr_values(data_dir, fetch_list=["text", "label"], encoding="utf-8")
    logging.info("data num = {}".format(len(text_list)))

    token_ids, token_type_ids = zip(*[tokenizer.encode(text) for text in text_list])

    label_ids = [label_encoder.transform(label_name) for label_name in label_list]

    data_list = list(zip(token_ids, token_type_ids, label_ids))

    train_text, test_text, train_data_list, test_data_list = \
        train_test_split(text_list, data_list,
                         test_size=test_size, random_state=random_state, shuffle=shuffle)
    logging.info("train num = {}".format(len(train_data_list)))
    logging.info("test num = {}".format(len(test_data_list)))

    train_dataset =  ClassificationDataset(train_data_list)
    test_dataset =  ClassificationDataset(test_data_list)

    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        )

    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        )

    example_num = 5
    logging.info(u"数据样例")
    for index, (text, (token_ids, _, label_id)) in enumerate(zip(
            train_text[:example_num],
            train_data_list[:example_num],
            )):
        label_name = label_encoder.inverse_transform(label_id)
        logging.info("example #{}:".format(index))
        logging.info("label: {}".format(label_name))
        logging.info("text: {}".format(text.encode("utf-8")))
        logging.info("token_ids: {}".format(token_ids))

    return train_dataloader, test_dataloader


class BertClassificationModel(ClassificationModel):
    def init_model(self, model_dir, tokenizer, keep_tokens):
        bert_model = BertForClassification.from_pretrained(
                model_dir,
                vocab_size=tokenizer.vocab_size,
                keep_tokens=keep_tokens)
        return bert_model


class TestBertClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        #vocab_path = "./state_dict/roberta_wwm/vocab.txt" # roberta模型字典的位置

        vocab_path = "./state_dict/bert_base_chinese/bert-base-chinese-vocab.txt" # roberta模型字典的位置

        TestBertClassification.tokenizer, TestBertClassification.keep_tokens = \
                BertTokenizer.load(vocab_path, simplified=True)

        dataset_root = "../dataset/"
        label_id_path = os.path.join(dataset_root, "classification_data/toutiao_news/class_id.txt")
        classification_data_dir = os.path.join(dataset_root, "classification_data/toutiao_news/toutiao_news_shrink.txt")

        TestBertClassification.label_encoder = LabelEncoder(label_id_path)

        TestBertClassification.train_dataloader, TestBertClassification.test_dataloader = create_train_test_dataset(
                classification_data_dir, TestBertClassification.tokenizer, TestBertClassification.label_encoder)

    def test_bert_classification(self):
        bert_config = {
                "model_dir": "./state_dict/bert_base_chinese/", # bert模型目录
                "vocab_size": TestBertClassification.tokenizer.vocab_size,
                "num_class": TestBertClassification.label_encoder.size(),
                "keep_tokens": TestBertClassification.keep_tokens,
                }

        model_dir = "./state_dict/bert_base_chinese/" # bert模型目录

        run_config = {
                "model_save_path": "./output/bert_class",
                "best_model_save_path": "./output/bert_class_best",
                "epochs": 2,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": False,
                }

        start_time = time.time()
        class BertClassificationModel(ClassificationModel):
            def init_model(self, model_dir, vocab_size, num_class, keep_tokens):
                model = BertForClassification.from_pretrained(
                        model_dir,
                        vocab_size=vocab_size,
                        num_class=num_class,
                        keep_tokens=keep_tokens,
                        )
                return model

        model = BertClassificationModel(**bert_config)
        best_score = model.train(
                TestBertClassification.train_dataloader,
                TestBertClassification.test_dataloader,
                label_encoder=TestBertClassification.label_encoder,
                **run_config)
        logging.info("best_score = {}.".format(best_score))


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestBertClassification("test_bert_classification"))
    runner = unittest.TextTestRunner()
    runner.run(suit)
