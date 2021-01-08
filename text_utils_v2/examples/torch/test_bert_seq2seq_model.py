#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_bert_seq2seq_model.py
Author:   zhanghao55@baidu.com
Date  :   21/01/05 19:29:27
Desc  :
"""

import os
import sys
import logging
import pandas as pd
import torch
import unittest
from torch.utils.data import Dataset, DataLoader


_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from process_data import process_origin_poetry

from text_utils.models.torch.base_model import BertSeq2seqModel, model_distributed
from text_utils.models.torch.nets.bert import BertForSeq2seq
from text_utils.tokenizers.bert_tokenizer import BertTokenizer
from text_utils.utils.data_io import get_file_name_list, write_to_file, get_data
from text_utils.utils.logger import init_log

init_log(stream_level=logging.INFO)


class PoetDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data_dir, tokenizer) :
        ## 一般init函数是加载所有数据
        super(PoetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.poet_info_list = self.gen_dataset(data_dir)

    def gen_dataset(self, data_dir):
        data_list = list()

        for line in get_data(data_dir):
            # 遍历文件夹中每首诗
            parts = line.strip("\n").split("\t")
            title = parts[0]
            poet = parts[3]

            # 处理
            # 标题取空格之前的
            title = title.split(" ")[0]

            # 根据诗的形式进行不同处理
            if len(poet) == 24 and (poet[5] == "，" or poet[5] == "。"):
                # 五言绝句
                title += "##" + "五言绝句"
            elif len(poet) == 32 and (poet[7] == "，" or poet[7] == "。"):
                # 七言绝句
                title += "##" + "七言绝句"
            elif len(poet) == 48 and (poet[5] == "，" or poet[5] == "。"):
                # 五言律诗
                title += "##" + "五言律诗"
            elif len(poet) == 64 and (poet[7] == "，" or poet[7] == "。"):
                # 七言律诗
                title += "##" + "七言律诗"
            else:
                # 不属于上述的跳过
                continue

            poet_ids, _ = self.tokenizer.encode(poet)
            # 诗中有未知字的跳过
            if self.tokenizer._token_unk_id in poet_ids:
                continue

            # 到这说明该诗符合要求
            data_list.append((title, poet))

            if len(data_list) > 2000:
                break

        logging.info("诗句共: " + str(len(data_list)) + "篇")
        return data_list


    def __getitem__(self, index):
        # 得到单个数据
        title, poet = self.poet_info_list[index]
        token_ids, token_type_ids = self.tokenizer.encode(title, poet)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):
        return len(self.poet_info_list)


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

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class BertPoemModel(BertSeq2seqModel):
    def init_model(self, model_dir, tokenizer=None, keep_tokens=None):
        bert_model = BertForSeq2seq.from_pretrained(
                model_dir,
                vocab_size=tokenizer.vocab_size,
                keep_tokens=keep_tokens)
        return bert_model


class TestSeq2seq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        #vocab_path = "./state_dict/roberta_wwm/vocab.txt" # roberta模型字典的位置

        vocab_path = "./state_dict/bert_base_chinese/bert-base-chinese-vocab.txt" # roberta模型字典的位置

        batch_size = 32

        dataset_root = "../dataset/"
        origin_poet_dir = os.path.join(dataset_root, "poetry")
        processed_poet_dir = os.path.join(dataset_root, "poetry_processed")

        TestSeq2seq.tokenizer, TestSeq2seq.keep_tokens = BertTokenizer.load(vocab_path, simplified=True)

        process_origin_poetry(origin_poet_dir, processed_poet_dir, overwrite=False)
        poet_dataset = PoetDataset(processed_poet_dir, TestSeq2seq.tokenizer)
        TestSeq2seq.dataloader =  DataLoader(
                poet_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                )

    def test_writing_poem_train(self):
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # roberta模型位置

        model_dir = "./state_dict/bert_base_chinese/" # bert模型目录

        run_config = {
                "model_save_path": "./output/bert_poem",
                "best_model_save_path": "./output/bert_poem_best",
                "epochs": 2,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": True,
                }

        model = BertPoemModel(model_dir, tokenizer=TestSeq2seq.tokenizer, keep_tokens=TestSeq2seq.keep_tokens)
        logging.info("device : {}".format(model.device))
        best_loss = model.train(TestSeq2seq.dataloader, TestSeq2seq.dataloader, **run_config)

        logging.info("best_loss = {}.".format(best_loss))

    def test_writing_poem(self):
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # roberta模型位置

        model_dir = "./state_dict/bert_base_chinese/" # bert模型目录

        run_config = {
                "model_save_path": "./output/bert_poem",
                "best_model_save_path": "./output/bert_poem_best",
                "epochs": 2,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": True,
                }

        model = BertPoemModel(model_dir, tokenizer=TestSeq2seq.tokenizer, keep_tokens=TestSeq2seq.keep_tokens)
        model.load_model(run_config["best_model_save_path"])
        test_data = [
                "北国风光##五言绝句",
                "题西林壁##七言绝句",
                "长安早春##五言律诗",
                "无题##五言绝句",
                "无题##五言律诗",
                "朱本常##五言律诗",
                "朱本常##五言绝句",
                "朱本常##七言绝句",
                "甘才钊##五言律诗",
                "甘才钊##五言绝句",
                "甘才钊##七言绝句",
                "考试太差了##五言绝句",
                "考试太差了##七言绝句",
                "回乡偶书##五言绝句",
                "回乡偶书##七言绝句",
                ]
        for text in test_data:
            logging.info(text)
            logging.info(model.generate(text, beam_size=3, device=model.device, is_poem=False))


if __name__ == '__main__':
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    #suit.addTest(TestSeq2seq("test_writing_poem_train"))
    suit.addTest(TestSeq2seq("test_writing_poem"))
    runner = unittest.TextTestRunner()
    runner.run(suit)


