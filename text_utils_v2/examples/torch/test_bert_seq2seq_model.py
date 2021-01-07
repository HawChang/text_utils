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
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)
from text_utils.utils.data_io import get_file_name_list, write_to_file, get_data
from text_utils.models.torch.base_model import BertSeq2seqModel, model_distributed
from text_utils.models.torch.seq2seq_model import BertSeq2seqNet

## 自动写诗的例子
import logging
import torch
from tqdm import tqdm
from torch.optim import Adam
import pandas as pd
import time
import unittest
from torch.utils.data import Dataset, DataLoader
from text_utils.tokenizers.bert_tokenizer import Tokenizer, load_chinese_base_vocab
#from text_utils.models.torch.utils import load_bert, load_model_params
from text_utils.utils.logger import init_log

init_log(stream_level=logging.INFO)


def process_origin_poetry(data_dir, dst_dir, overwrite=False):
    if not os.path.isdir(dst_dir):
        logging.debug("create data dir: {}".format(dst_dir))
        os.mkdir(dst_dir)

    def gen_poet_iter(data_path):
        df = pd.read_csv(data_path)
        for index, row in df.iterrows():
            try:
                title = row[0].replace("\t", "\\t")
                dynasty = row[1].replace("\t", "\\t")
                author = row[2].replace("\t", "\\t")
                poet = row[3].replace("\t", "\\t")
                yield "\t".join([title, dynasty, author, poet])
            except AttributeError as e:
                logging.warning("parse poet fail at line #{}".format(index + 1))

    for data_path in get_file_name_list(data_dir):
        file_name = data_path[data_path.rfind("/")+1:]
        # 隐藏文件 或 不为csv的文件 跳过
        if file_name.startswith(".") or (not file_name.endswith("csv")):
            continue
        logging.debug("process: {}".format(data_path))
        dst_path = os.path.join(dst_dir, file_name)
        if os.path.exists(dst_path):
            if not overwrite:
                logging.debug("{} already processed, skip.".format(file_name))
                continue
        write_to_file(gen_poet_iter(data_path), dst_path)


class PoetDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data_dir, word2idx) :
        ## 一般init函数是加载所有数据
        super(PoetDataset, self).__init__()
        self.word2idx = word2idx
        self.tokenizer = Tokenizer(word2idx)
        self.poet_info_list = self.gen_dataset(data_dir)

    def gen_dataset(self, data_dir):
        data_list = list()

        tokenizer = Tokenizer(self.word2idx)
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

            poet_ids = tokenizer.encode(poet)[0]
            # 诗中有未知字的跳过
            if self.word2idx["[UNK]"] in poet_ids:
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


class TestSeq2seq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        #vocab_path = "./state_dict/roberta_wwm/vocab.txt" # roberta模型字典的位置

        vocab_path = "./state_dict/bert_base_chinese/bert-base-chinese-vocab.txt" # roberta模型字典的位置

        batch_size = 32
        #TestSeq2seq.lr = 1e-5
        #TestSeq2seq.epochs = 50

        dataset_root = "../dataset/"
        origin_poet_dir = os.path.join(dataset_root, "poetry")
        processed_poet_dir = os.path.join(dataset_root, "poetry_processed")

        TestSeq2seq.word2idx, TestSeq2seq.keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)

        process_origin_poetry(origin_poet_dir, processed_poet_dir, overwrite=False)
        poet_dataset = PoetDataset(processed_poet_dir, TestSeq2seq.word2idx)
        TestSeq2seq.dataloader =  DataLoader(
                poet_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                )

    def test_writing_poem_train(self):
        #model_name = "roberta" # 选择模型名字
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # roberta模型位置
        ##recent_model_path = "./output/roberta_model_poem.bin" # 用于把已经训练好的模型继续训练
        #model_save_path = "./output/roberta_model_poem.bin"

        model_name = "bert" # 选择模型名字
        model_path = "./state_dict/bert_base_chinese/bert-base-chinese-pytorch_model.bin" # roberta模型位置
        model_dir = "./state_dict/bert_base_chinese/" # bert模型目录
        #recent_model_path = "./output/bert_model_poem.bin" # 用于把已经训练好的模型继续训练
        #model_save_path = "./output/bert_model_poem.bin"

        run_config = {
                "model_save_path": "./output/bert_poem",
                "best_model_save_path": "./output/bert_poem_best",
                "epochs": 2,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": True,
                }
        tokenizer = Tokenizer(TestSeq2seq.word2idx)

        class BertPoemModel(BertSeq2seqModel):
            def init_model(self, model_dir):
                bert_model = BertSeq2seqNet.from_pretrained(
                        model_dir,
                        tokenizer=tokenizer,
                        vocab_size=len(TestSeq2seq.word2idx),
                        keep_tokens=TestSeq2seq.keep_tokens)
                #load_bert(word2idx, model_name=model_name)
                # 加载预训练的模型参数
                #load_model_params(bert_model, model_path, keep_tokens=TestSeq2seq.keep_tokens)
                return bert_model

        model = BertPoemModel(model_dir)
        logging.info("device : {}".format(model.device))
        best_loss = model.train(TestSeq2seq.dataloader, TestSeq2seq.dataloader, **run_config)

        logging.info("best_loss = {}.".format(best_loss))


if __name__ == '__main__':
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestSeq2seq("test_writing_poem_train"))
    runner = unittest.TextTestRunner()
    runner.run(suit)


