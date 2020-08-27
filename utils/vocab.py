#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   vocab.py
Author:   zhanghao55@baidu.com
Date  :   20/08/11 14:04:07
Desc  :   
"""

import logging
from collections import defaultdict
from data_io import write_to_file
from data_io import read_from_file


class Vocab(object):
    def __init__(self):
        """初始化
        """
        self.vocab_id = None
        self.id_vocab = None
        self.oov = None

    def fit(self, tokens_list, oov="<unk>", min_df=2):
        """建立vocab
        """
        self.vocab_id = dict()
        if oov is not None:
            self.oov = oov
            self.vocab_id[oov] = len(self.vocab_id)
            self.oov_id = self.vocab_id[oov]

        vocab_count = defaultdict(int)
        for tokens in tokens_list:
            for token in tokens:
                vocab_count[token] += 1

        vocab_count = sorted(vocab_count.items(), key=lambda x:x[1], reverse=True)
        logging.debug("vocab frequency top 10:")
        for vocab, count in vocab_count[:10]:
            logging.debug("%s: %d" % (vocab.encode("gb18030"), count))

        for vocab, count in vocab_count:
            # 如果小于min_df 则后面的都小于 直接跳出
            if count < min_df:
                break
            self.vocab_id[vocab] = len(self.vocab_id)

        self.id_vocab = {v:k for k, v in self.vocab_id.items()}

    def transform(self, tokens_list):
        """转换
        """
        if self.vocab_id is None:
            raise ValueError("vocab is None, need to fit vocab before transform")

        if self.oov is None:
            return [[self.vocab_id[token] for token in tokens] for tokens in tokens_list]
        else:
            return [[self.vocab_id.get(token, self.oov_id) for token in tokens] for tokens in tokens_list]

    def fit_transform(self, tokens_list, **kwargs):
        """建立vocab并返回转换后的列表
        """
        self.fit(tokens_list, **kwargs)
        return self.transform(tokens_list)

    def inverse_transform(self, token_ids_list, oov="<unk>"):
        """id转为token
        """
        if self.id_vocab is None:
            raise ValueError("vocab is None, need to fit vocab before inverse transform")

        return [[self.id_vocab.get(token_id, oov) for token_id in token_ids] for token_ids in token_ids_list]

    def save(self, save_path):
        """保存
        """
        def gen_vocab_iter():
            for v, v_id in sorted(self.vocab_id.items(), key=lambda x: x[1]):
                yield "\t".join([v, str(v_id)])

        write_to_file(gen_vocab_iter(), save_path)

    def load(self, load_path, oov="<unk>"):
        """加载
        """
        self.vocab_id = dict()
        vocab_id_list = read_from_file(load_path, read_func=lambda x: x.strip("\n").split("\t"))
        for v, v_id in vocab_id_list:
            v_id = int(v_id)
            if v == oov:
                self.oov = v
                self.oov_id = v_id
            self.vocab_id[v] = v_id

        self.id_vocab = {v:k for k, v in self.vocab_id.items()}

    def size(self):
        return len(self.vocab_id)

    def __len__(self):
        return len(self.vocab_id)


if __name__ == "__main__":
    pass


