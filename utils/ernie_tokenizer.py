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

_max_input_chars_per_word = 100

def _wordpiece(token, vocab, unk_token, prefix='##', sentencepiece_prefix=''):
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
    @classmethod
    def from_pretrained(cls, vocab_path, **kwargs):
        if not os.path.exists(vocab_path):
            raise ValueError('no vocab file in pretrain dir: %s' % vocab_path)
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
        pat_str += r'([a-zA-Z0-9]+|\S)'
        logging.debug('regex: %s' % pat_str)
        self.pat = re.compile(pat_str)
        self.encoding = encoding

    def tokenize(self, text):
        if len(text) == 0:
            return []
        if six.PY3 and not isinstance(text, six.string_types):
            text = text.decode(self.encoding)
        if six.PY2 and isinstance(text, str):
            text = text.decode(self.encoding)

        res = []
        for match in self.pat.finditer(text):
            match_group = match.group(0)
            if match.groups()[-1]:
                if self.lower:
                    match_group = match_group.lower()
                words, _ = _wordpiece(match_group, vocab=self.vocab, unk_token=self.unk_token, prefix=self.prefix, sentencepiece_prefix=self.sentencepiece_prefix)
            else:
                words = [match_group]
            res += words
        return res

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def truncate(self, id1, id2, seqlen):
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
        text_id = np.array(self.convert_tokens_to_ids(self.tokenize(text)), dtype=np.int64)
        text_id_type = np.zeros_like(text_id, dtype=np.int64)
        if pair is not None:
            pair_id = np.array(self.convert_tokens_to_ids(self.tokenize(pair)), dtype=np.int64)
        else:
            pair_id = []
        if truncate_to is not None:
            text_id, pair_id = self.truncate(text_id, [] if pair_id is None else pair_id, truncate_to)

        ret_id, ret_id_type = self.build_for_ernie(text_id, pair_id)
        return ret_id, ret_id_type


if __name__ == "__main__":
    pass


