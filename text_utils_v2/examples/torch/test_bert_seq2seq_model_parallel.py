#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_bert_seq2seq_model_parallel.py
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

## �Զ�дʫ������
import logging
import torch
from tqdm import tqdm
from torch.optim import Adam
import pandas as pd
import time
import unittest
from torch.utils.data import Dataset, DataLoader
from text_utils.tokenizers.bert_tokenizer import Tokenizer, load_chinese_base_vocab
from text_utils.models.torch.utils import load_bert, load_model_params
from text_utils.utils.logger import init_log

from torch.utils.data.distributed import DistributedSampler
torch.distributed.init_process_group(backend="nccl")
LOCAL_RANK = torch.distributed.get_rank()

logging_level = logging.INFO if LOCAL_RANK == 0 else logging.WARNING
init_log(stream_level=logging_level)


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
        # �����ļ� �� ��Ϊcsv���ļ� ����
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
    ����ض����ݼ�������һ����ص�ȡ���ݵķ�ʽ
    """
    def __init__(self, data_dir, word2idx) :
        ## һ��init�����Ǽ�����������
        super(PoetDataset, self).__init__()
        self.word2idx = word2idx
        self.tokenizer = Tokenizer(word2idx)
        self.poet_info_list = self.gen_dataset(data_dir)

    def gen_dataset(self, data_dir):
        data_list = list()

        tokenizer = Tokenizer(self.word2idx)
        for line in get_data(data_dir):
            # �����ļ�����ÿ��ʫ
            parts = line.strip("\n").split("\t")
            title = parts[0]
            poet = parts[3]

            # ����
            # ����ȡ�ո�֮ǰ��
            title = title.split(" ")[0]

            # ����ʫ����ʽ���в�ͬ����
            if len(poet) == 24 and (poet[5] == "��" or poet[5] == "��"):
                # ���Ծ���
                title += "##" + "���Ծ���"
            elif len(poet) == 32 and (poet[7] == "��" or poet[7] == "��"):
                # ���Ծ���
                title += "##" + "���Ծ���"
            elif len(poet) == 48 and (poet[5] == "��" or poet[5] == "��"):
                # ������ʫ
                title += "##" + "������ʫ"
            elif len(poet) == 64 and (poet[7] == "��" or poet[7] == "��"):
                # ������ʫ
                title += "##" + "������ʫ"
            else:
                # ����������������
                continue

            poet_ids = tokenizer.encode(poet)[0]
            # ʫ����δ֪�ֵ�����
            if self.word2idx["[UNK]"] in poet_ids:
                continue

            # ����˵����ʫ����Ҫ��
            data_list.append((title, poet))

            if len(data_list) > 2000:
                break

        logging.info("ʫ�乲: " + str(len(data_list)) + "ƪ")
        return data_list


    def __getitem__(self, index):
        # �õ���������
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
    ��̬padding�� batchΪһ����sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad ����
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

        #vocab_path = "./state_dict/roberta_wwm/vocab.txt" # robertaģ���ֵ��λ��

        vocab_path = "./state_dict/bert_base_chinese/bert-base-chinese-vocab.txt" # robertaģ���ֵ��λ��

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
                collate_fn=collate_fn,
                sampler=DistributedSampler(poet_dataset),
                )

    def test_writing_poem_train(self):
        #model_name = "roberta" # ѡ��ģ������
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # robertaģ��λ��
        ##recent_model_path = "./output/roberta_model_poem.bin" # ���ڰ��Ѿ�ѵ���õ�ģ�ͼ���ѵ��
        #model_save_path = "./output/roberta_model_poem.bin"

        model_name = "bert" # ѡ��ģ������
        model_path = "./state_dict/bert_base_chinese/bert-base-chinese-pytorch_model.bin" # robertaģ��λ��
        #recent_model_path = "./output/bert_model_poem.bin" # ���ڰ��Ѿ�ѵ���õ�ģ�ͼ���ѵ��
        #model_save_path = "./output/bert_model_poem.bin"

        run_config = {
                "model_save_path": "./output/bert_poem",
                "best_model_save_path": "./output/bert_poem_best",
                "epochs": 2,
                "print_step": 100,
                "learning_rate": 5e-5,
                "load_best_model": True,
                }

        class BertPoemModel(BertSeq2seqModel):
            @model_distributed(find_unused_parameters=True)
            def init_model(self, word2idx, model_name, model_path):
                bert_model = load_bert(word2idx, model_name=model_name)
                # ����Ԥѵ����ģ�Ͳ���
                load_model_params(bert_model, model_path, keep_tokens=TestSeq2seq.keep_tokens)
                return bert_model

        model = BertPoemModel(TestSeq2seq.word2idx, model_name, model_path)
        logging.info("device : {}".format(model.device))
        best_loss = model.train(TestSeq2seq.dataloader, TestSeq2seq.dataloader, **run_config)

        logging.info("best_loss = {}.".format(best_loss))

    #def gen_poem(self, model):
    #    model.eval()
    #    test_data = ["�������##���Ծ���", "�����ֱ�##���Ծ���", "�����紺##������ʫ"]
    #    for text in test_data:
    #        logging.debug(text)
    #        logging.debug(model.generate(text, beam_size=3,device=TestSeq2seq.device, is_poem=True))
    #    model.train()


if __name__ == '__main__':
    #process_origin_poetry("../dataset/poetry/", "../dataset/poetry_processed", overwrite=False)
    # �������в�������
    #unittest.main()

    # ����ָ����������
    # ������Լ�
    suit = unittest.TestSuite()
    suit.addTest(TestSeq2seq("test_writing_poem_train"))
    runner = unittest.TextTestRunner()
    runner.run(suit)


