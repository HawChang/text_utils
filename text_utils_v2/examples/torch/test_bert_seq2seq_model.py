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
import numpy as np
import torch
import unittest
from torch.utils.data import Dataset, DataLoader

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from process_data import process_origin_poetry

from text_utils.models.torch.base_model import BertSeq2seqModel, model_distributed
from text_utils.models.torch.nets.bert import BertForSeq2seq
from text_utils.tokenizers.bert_tokenizer import BertTokenizer
from text_utils.utils.data_io import get_data
from text_utils.utils.logger import init_log

init_log(stream_level=logging.INFO)


class PoetDataset(Dataset):
    """
    ����ض����ݼ�������һ����ص�ȡ���ݵķ�ʽ
    """
    def __init__(self, data_dir, tokenizer) :
        ## һ��init�����Ǽ�����������
        super(PoetDataset, self).__init__()
        self.tokenizer = tokenizer
        # dataloader�е�������numpy����
        # ���issue: https://github.com/pytorch/pytorch/issues/13246
        self.poet_info_list = np.array(self.gen_dataset(data_dir))

    def gen_dataset(self, data_dir):
        data_list = list()

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

            poet_ids, _ = self.tokenizer.encode(poet)
            # ʫ����δ֪�ֵ�����
            if self.tokenizer._token_unk_id in poet_ids:
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


class BertPoemModel(BertSeq2seqModel):
    def init_model(self, model_dir, tokenizer, keep_tokens):
        bert_model = BertForSeq2seq.from_pretrained(
                model_dir,
                vocab_size=tokenizer.vocab_size,
                keep_tokens=keep_tokens)
        return bert_model


class TestSeq2seq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        #vocab_path = "./state_dict/roberta_wwm/vocab.txt" # robertaģ���ֵ��λ��

        vocab_path = "./state_dict/bert_base_chinese/bert-base-chinese-vocab.txt" # robertaģ���ֵ��λ��

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
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # robertaģ��λ��

        model_dir = "./state_dict/bert_base_chinese/" # bertģ��Ŀ¼

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
        #model_path = "./state_dict/roberta_wwm/pytorch_model.bin" # robertaģ��λ��

        model_dir = "./state_dict/bert_base_chinese/" # bertģ��Ŀ¼

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
                "�������##���Ծ���",
                "�����ֱ�##���Ծ���",
                "�����紺##������ʫ",
                "����##���Ծ���",
                "����##������ʫ",
                "�챾��##������ʫ",
                "�챾��##���Ծ���",
                "�챾��##���Ծ���",
                "�ʲ���##������ʫ",
                "�ʲ���##���Ծ���",
                "�ʲ���##���Ծ���",
                "����̫����##���Ծ���",
                "����̫����##���Ծ���",
                "����ż��##���Ծ���",
                "����ż��##���Ծ���",
                ]
        for text in test_data:
            logging.info(text)
            logging.info(model.generate(text, beam_size=3, device=model.device, is_poem=True))


if __name__ == '__main__':
    # �������в�������
    #unittest.main()

    # ����ָ����������
    # ������Լ�
    suit = unittest.TestSuite()
    suit.addTest(TestSeq2seq("test_writing_poem_train"))
    suit.addTest(TestSeq2seq("test_writing_poem"))
    runner = unittest.TextTestRunner()
    runner.run(suit)


