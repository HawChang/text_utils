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
import time

from text_utils.utils import pygtrie


class ErnieTokenizer(object):
    """用于tokenize得到ernie模型接受的输入的工具类
    """
    @classmethod
    def load(cls, vocab_path, encoding='utf-8', **kwargs):
        """初始化
        """
        if not os.path.exists(vocab_path):
            raise ValueError('no vocab file in vocab path: %s' % vocab_path)

        vocab_dict = dict()
        with codecs.open(vocab_path, "r", encoding) as rf:
            for index, line in enumerate(rf):
                vocab_dict[line.strip().split('\t')[0]] = index
        #vocab_dict = {j.strip().split('\t')[0]: i for i, j in enumerate(open(vocab_path).readlines())}
        tokenizer_instance = cls(vocab_dict, **kwargs)
        return tokenizer_instance

    def __init__(self,
            vocab_dict,
            unk_token='[UNK]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            pad_token='[PAD]',
            mask_token='[MASK]',
            special_token_list=None,
            lower=True,
            wordpiece_prefix='##',
            sentencepiece_prefix='',
            max_input_chars_per_word=100,
            ):
        if not isinstance(vocab_dict, dict):
            raise ValueError('expect `vocab_dict` to be instance of dict, got %s' % type(vocab_dict))
        self.vocab_dict = vocab_dict
        self.vocab_tree = pygtrie.CharTrie()
        for key, value in self.vocab_dict.items():
            self.vocab_tree[key] = value

        self.lower = lower
        self.wordpiece_prefix = wordpiece_prefix
        self.sentencepiece_prefix = sentencepiece_prefix
        self.max_input_chars_per_word = max_input_chars_per_word

        self.pad_id = self.vocab_dict[pad_token]
        self.cls_id = cls_token and self.vocab_dict[cls_token]
        self.sep_id = sep_token and self.vocab_dict[sep_token]
        self.unk_id = unk_token and self.vocab_dict[unk_token]
        self.mask_id = mask_token and self.vocab_dict[mask_token]
        self.unk_token = unk_token

        if special_token_list is None:
            special_token_list = []
        special_tokens = {pad_token, cls_token, sep_token, unk_token, mask_token} | set(special_token_list)
        token_pattern = ''
        for t in special_tokens:
            if t is None:
                continue
            token_pattern += '(%s)|' % re.escape(t)
        token_pattern += r'([^a-zA-Z0-9 \t\n\r\f])|([a-zA-Z0-9]+)'
        logging.debug('regex: %s' % token_pattern)
        # pattern: (\[CLS\])|(\[SEP\])|(\[PAD\])|(\[MASK\])|(\[UNK\])|([^a-zA-Z0-9 \t\n\r\f])|([a-zA-Z0-9]+)
        self.pat = re.compile(token_pattern)

    def tokenize(self, text):
        """tokenizer
        [IN]  text: string, 待tokenize字符串
        [OUT] res: list[id], tokenize后得到的数组
        """
        if len(text) == 0:
            return []

        res = []
        for match in self.pat.finditer(text):
            # 取当前匹配的结果
            cur_token = match.group(0)
            # groups是七元组，前五个是特殊标记，第六个是单个的除字母、数字和空白的字符，第七个是连续的字母或数字
            if match.groups()[-1]:
                # 如果匹配到的是最后一个 则是连续的字母或数字
                if self.lower:
                    # 按要求转小写
                    cur_token = cur_token.lower()
                words, _ = self.wordpiece(cur_token)
            else:
                words = [cur_token]
            res += words
        #print("tokenize res: [{}]".format(",".join(res).encode("utf-8")))
        return res

    def wordpiece(self, token):
        """wordpiece: helloworld => [hello, ##world]
        """
        token_lenth = len(token)
        if token_lenth > self.max_input_chars_per_word:
            return [self.unk_token], [(0, token_lenth)]

        is_bad = False
        start = 0
        sub_token_list = []
        sub_pos_list = []
        while start < len(token):
            # 各位置的piece有不同的前缀
            if start == 0:
                cur_prefix = self.sentencepiece_prefix
            else:
                cur_prefix = self.wordpiece_prefix

            cur_match_text = cur_prefix + token[start:]
            #print("cur_match_text: {}".format(cur_match_text.encode("utf-8")))
            key, value = self.vocab_tree.longest_prefix(cur_match_text)

            # 如果当前片段不能匹配 则失败 置为UNK
            if key is None:
                is_bad = True
                break

            #print("key: {}".format(key.encode("utf-8")))
            # 匹配上则分为一个part
            sub_token_list.append(key)

            # 计算该part位置 以及新start的位置
            end = start + len(key) - len(cur_prefix)
            sub_pos_list.append((start, end))
            start = end

        if is_bad:
            return [unk_token], [(0, token_lenth)]
        else:
            return sub_token_list, sub_pos_list

    def convert_tokens_to_ids(self, tokens):
        """tokens -> ids
        """
        return [self.vocab_dict.get(t, self.unk_id) for t in tokens]

    def truncate(self, id1, id2, seqlen):
        """截断、连接id1和id2
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
        """build sentence type id, add [CLS] [SEP]
        """
        ret_id = np.concatenate([[self.cls_id], text_id, [self.sep_id]], 0)
        ret_id_type = np.zeros_like(ret_id, dtype=np.int64)
        #ret_id_type = np.concatenate([[0], text_id_type, [0]], 0)

        if len(pair_id):
            pair_id_type = np.ones_like(pair_id, dtype=np.int64)
            ret_id = np.concatenate([ret_id, pair_id, [self.sep_id]], 0)
            ret_id_type = np.concatenate([ret_id_type, pair_id_type, [1]], 0)

        return ret_id, ret_id_type

    def encode(self, text, pair=None, truncate_to=None):
        """对text编码
        """
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
            #text_id, pair_id = self.truncate(text_id, [] if pair_id is None else pair_id, truncate_to)
            # pair_id不会为None
            text_id, pair_id = self.truncate(text_id, pair_id, truncate_to)
        #other_time = time.time() - start_time

        #start_time = time.time()
        ret_id, ret_id_type = self.build_for_ernie(text_id, pair_id)
        #build_for_ernie_time = time.time() - start_time

        #return ret_id, ret_id_type, tokenize_time, convert_token_time , other_time, build_for_ernie_time
        return ret_id, ret_id_type

    def transform(self, text_list):
        """
        """
        ret_id_list = list()
        start_time = time.time()
        for text in text_list:
            ret_id = self.encode(text)[0]
            ret_id_list.append(ret_id)
        cost_time = time.time() - start_time
        data_num = len(text_list)
        speed = data_num / cost_time
        logging.info("transform data num = {}, cost time = {:.4f}, " \
                "speed = {:.2f}/s".format(data_num, cost_time, speed))
        return ret_id_list

    def size(self):
        """
        """
        return len(self.vocab_dict)


if __name__ == "__main__":
    import time

    VOCAB_PATH = "model/vocab.txt"
    # 获得数据
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

    # 展示部分数据
    logging.info("数据样例：")
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
