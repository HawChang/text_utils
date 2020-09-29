#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   ernie_tokenizer.py
Author:   zhanghao55@baidu.com
Date  :   20/08/14 12:52:40
Desc  :   
"""

import codecs
import logging
import numpy as np
import os
import re
import six
import time
import pygtrie

_max_input_chars_per_word = 100

def _wordpiece(token, vocab_tree, unk_token, prefix='##', sentencepiece_prefix=''):
    """ wordpiece: helloworld => [hello, ##world] """
    chars = token
    token_lenth = len(token)
    if token_lenth > _max_input_chars_per_word:
        return [unk_token], [(0, token_lenth)]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(token):
        # ��λ�õ�piece�в�ͬ��ǰ׺
        if start == 0:
            cur_prefix = sentencepiece_prefix
        else:
            cur_prefix = prefix

        cur_match_text = cur_prefix + token[start:]
        #print("cur_match_text: {}".format(cur_match_text.encode("utf-8")))
        key, value = vocab_tree.longest_prefix(cur_match_text)

        # �����ǰƬ�β���ƥ�� ��ʧ�� ��ΪUNK
        if key is None:
            is_bad = True
            break

        #print("key: {}".format(key.encode("utf-8")))
        # ƥ�������Ϊһ��part
        sub_tokens.append(key)

        # �����partλ�� �Լ���start��λ��
        end = start + len(key) - len(cur_prefix)
        sub_pos.append((start, end))
        start = end

    if is_bad:
        return [unk_token], [(0, token_lenth)]
    else:
        return sub_tokens, sub_pos

def _wordpiece_old(token, vocab, unk_token, prefix='##', sentencepiece_prefix=''):
    """ wordpiece: helloworld => [hello, ##world] """
    chars = list(token)
    if len(chars) > _max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0:
                substr = sentencepiece_prefix + substr
            if start > 0:
                substr = prefix + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        return [unk_token], [(0, len(chars))]
    else:
        return sub_tokens, sub_pos

class ErnieTokenizer(object):
    """����tokenize�õ�ernieģ�ͽ��ܵ�����Ĺ�����
    """
    @classmethod
    def from_pretrained(cls, vocab_path, **kwargs):
        """��ʼ��
        """
        if not os.path.exists(vocab_path):
            raise ValueError('no vocab file in vocab path: %s' % vocab_path)
        #vocab_tree = pygtrie.CharTrie()
        vocab_dict = dict()
        with codecs.open(vocab_path, "r", "utf-8") as rf:
            for index, line in enumerate(rf):
                vocab_dict[line.strip().split('\t')[0]] = index
        #vocab_dict = {j.strip().split('\t')[0]: i for i, j in enumerate(open(vocab_path).readlines())}
        t = cls(vocab_dict, **kwargs)
        return t

    def __init__(self,
            vocab,
            unk_token='[UNK]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            pad_token='[PAD]',
            mask_token='[MASK]',
            wordpiece_prefix='##',
            sentencepiece_prefix='',
            lower=True,
            encoding='utf8',
            special_token_list=[]):
        if not isinstance(vocab, dict):
            raise ValueError('expect `vocab` to be instance of dict, got %s' % type(vocab))
        self.vocab = vocab
        self.vocab_tree = pygtrie.CharTrie()
        for key, value in self.vocab.items():
            self.vocab_tree[key] = value

        self.lower = lower
        self.prefix = wordpiece_prefix
        self.sentencepiece_prefix = sentencepiece_prefix
        self.pad_id = self.vocab[pad_token]
        self.cls_id = cls_token and self.vocab[cls_token]
        self.sep_id = sep_token and self.vocab[sep_token]
        self.unk_id = unk_token and self.vocab[unk_token]
        self.mask_id = mask_token and self.vocab[mask_token]
        self.unk_token = unk_token
        special_tokens = {pad_token, cls_token, sep_token, unk_token, mask_token} | set(special_token_list)
        pat_str = ''
        for t in special_tokens:
            if t is None:
                continue
            pat_str += '(%s)|' % re.escape(t)
        pat_str += r'([^a-zA-Z0-9 \t\n\r\f])|([a-zA-Z0-9]+)'
        logging.debug('regex: %s' % pat_str)
        # pattern: (\[CLS\])|(\[SEP\])|(\[PAD\])|(\[MASK\])|(\[UNK\])|([^a-zA-Z0-9 \t\n\r\f])|([a-zA-Z0-9]+)
        self.pat = re.compile(pat_str)
        self.encoding = encoding

    def tokenize(self, text):
        """tokenizer
        [IN]  text: string, ��tokenize�ַ���
        [OUT] res: list[id], tokenize��õ�������
        """
        #print("text: {}".format(text.encode("utf-8")))
        if len(text) == 0:
            return []

        res = []
        for match in self.pat.finditer(text):
            # ȡ��ǰƥ��Ľ��
            match_group = match.group(0)
            # groups����Ԫ�飬ǰ����������ǣ��������ǵ����ĳ���ĸ�����ֺͿհ׵��ַ������߸�����������ĸ������
            if match.groups()[-1]:
                # ���ƥ�䵽�������һ�� ������������ĸ������
                if self.lower:
                    # ��Ҫ��תСд
                    match_group = match_group.lower()
                words, _ = _wordpiece(match_group, vocab_tree=self.vocab_tree, unk_token=self.unk_token,
                        prefix=self.prefix, sentencepiece_prefix=self.sentencepiece_prefix)
            else:
                words = [match_group]
            res += words
        #print("tokenize res: [{}]".format(",".join(res).encode("utf-8")))
        return res

    def convert_tokens_to_ids(self, tokens):
        """tokens -> ids
        """
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def truncate(self, id1, id2, seqlen):
        """�ضϡ�����id1��id2
        """
        len1 = len(id1)
        len2 = len(id2)
        half = seqlen // 2
        if len1 > len2:
            len1_truncated, len2_truncated = max(half, seqlen - len2), min(half, len2)
        else:
            len1_truncated, len2_truncated = min(half, seqlen - len1), max(half, seqlen - len1)
        return id1[: len1_truncated], id2[: len2_truncated]

    def build_for_ernie(self, text_id, pair_id=[]):
        """build sentence type id, add [CLS] [SEP]"""
        text_id_type = np.zeros_like(text_id, dtype=np.int64)
        ret_id = np.concatenate([[self.cls_id], text_id, [self.sep_id]], 0)
        ret_id_type = np.concatenate([[0], text_id_type, [0]], 0)

        if len(pair_id):
            pair_id_type = np.ones_like(pair_id, dtype=np.int64)
            ret_id = np.concatenate([ret_id, pair_id, [self.sep_id]], 0)
            ret_id_type = np.concatenate([ret_id_type, pair_id_type, [1]], 0)
        return ret_id, ret_id_type

    def encode(self, text, pair=None, truncate_to=None):
        """��text����
        """
        #print("text: {}".format(text.encode("utf-8")))
        #start_time = time.time()
        tokenize_res = self.tokenize(text)
        #tokenize_time = time.time() - start_time

        #start_time = time.time()
        token_ids = self.convert_tokens_to_ids(tokenize_res)
        #convert_token_time = time.time() - start_time

        #start_time = time.time()
        text_id = np.array(token_ids, dtype=np.int64)
        text_id_type = np.zeros_like(text_id, dtype=np.int64)
        if pair is not None:
            pair_id = np.array(self.convert_tokens_to_ids(self.tokenize(pair)), dtype=np.int64)
        else:
            pair_id = []
        if truncate_to is not None:
            text_id, pair_id = self.truncate(text_id, [] if pair_id is None else pair_id, truncate_to)
        #other_time = time.time() - start_time

        #start_time = time.time()
        ret_id, ret_id_type = self.build_for_ernie(text_id, pair_id)
        #build_for_ernie_time = time.time() - start_time
        #print("re_id: {}".format(ret_id))

        #return ret_id, ret_id_type, tokenize_time, convert_token_time , other_time, build_for_ernie_time
        return ret_id, ret_id_type


if __name__ == "__main__":

    VOCAB_PATH = "model/vocab.txt"
    # �������
    from data_io import get_attr_values
    DATA_DIR="data/zui_mark_0812.txt"
    DATA_DIR="data/zui_eval_res_20200722.txt"
    DATA_DIR="data/tuling_sample_50000"
    DATA_DIR="data/leak_online.txt"
    #DATA_DIR="lo"
    #DATA_DIR="data/zhuxiaodong_sample_20200807_09.txt"
    #DATA_DIR="data/sample_200.txt"
    encoding="gb18030"
    data_list = get_attr_values(
            DATA_DIR,
            fetch_list=["text"],
            encoding=encoding,
            )

    data_list = list(data_list[0])[:]
    logging.info("data num = {}".format(len(data_list)))

    # չʾ��������
    logging.info("����������")
    for i in range(5):
        logging.info(data_list[i].encode("utf-8"))

    tokenizer = ErnieTokenizer.from_pretrained(VOCAB_PATH)
    #data_ids = [tokenizer.encode(x)[0] for x in data_list[:1000]]
    #data_ids = [tokenizer.encode(x)[0] for x in data_list[39:10000]]
    tokenize_time = 0
    convert_token_time = 0
    other_time = 0
    build_for_ernie_time = 0

    data_ids = list()
    #for x in data_list[39:10000]:
    encode_time = 0
    start_time = time.time()
    for x in data_list:
        cur_start_time = time.time()
        #cur_ids, _, cur_tokenize_time, cur_convert_token_time, cur_other_time, cur_build_for_ernie_time = tokenizer.encode(x)
        cur_ids, _ = tokenizer.encode(x)
        encode_time += time.time() - cur_start_time
        #tokenize_time += cur_tokenize_time
        #convert_token_time += cur_convert_token_time
        #other_time += cur_other_time
        #build_for_ernie_time = cur_build_for_ernie_time
        data_ids.append(cur_ids)
    #print("data_ids: {}".format(data_ids))
    print("total_time: {}".format(time.time() - start_time))
    #print("tokenize_time: {}".format(tokenize_time))
    #print("convert_token_time: {}".format(convert_token_time))
    #print("other_time: {}".format(other_time))
    #print("build_for_ernie_time: {}".format(build_for_ernie_time))
    print("encode_time: {}".format(encode_time))
