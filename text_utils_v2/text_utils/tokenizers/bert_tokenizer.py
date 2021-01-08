#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File  :   bert_tokenizer.py
Author:   zhanghao55@baidu.com
Date  :   21/01/08 12:07:05
Desc  :
"""

import logging
import unicodedata
from typing import List
import re

from text_utils.utils.data_io import write_to_file


class BasicTokenizer(object):

    @classmethod
    def load(cls, vocab_path, simplified=True, special_mark=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]):
        token_dict, keep_tokens = cls.load_chinese_base_vocab(vocab_path, simplified, special_mark)
        # 令vocab_origin_id为该vocab在vocab_path中的行数
        # token_dict[vocab] = vocab_id
        # 此时，vocab_id不是vocab_origin_id，而是vocab在预训练数据中的id（即vocab_origin_id）在keep_tokens的索引
        # 即 keep_tokens[token_dict[vocab]] = vocab_origin_id
        # 加载预训练数据时，embedding会根据keep_tokens而改变id 改变后embedding中vocab的id即为vocab_id，而不是vocab_origin_id
        tokenizer = cls(token_dict)
        # tokenizer不会用到keep_tokens 但模型加载预训练模型时需要
        return tokenizer, keep_tokens

    @classmethod
    def load_chinese_base_vocab(cls, vocab_path, simplified=False, special_mark=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]):
        """加载官方中文bert模型字典
        simplified: 是否简化词典
        """

        token_dict = dict()
        with open(vocab_path, "r", encoding="utf-8") as rf:
            for index, line in enumerate(rf):
                token_dict[line.strip("\n")] = index

        if simplified:
            new_token_dict = dict()
            keep_tokens = list()
            for m in special_mark:
                new_token_dict[m] = len(new_token_dict)
                keep_tokens.append(token_dict[m])

            for cur_token, _ in sorted(token_dict.items(), key=lambda x: x[1]):
                if cur_token not in new_token_dict:
                    keep = True
                    if len(cur_token) > 1:
                        for c in cls.stem(cur_token):
                            # 如果长度大于1的是token中包含cjk或标点 则跳过
                            if cls._is_cjk_character(c) or cls._is_punctuation(c):
                                keep = False
                                break
                    if keep:
                        new_token_dict[cur_token] = len(new_token_dict)
                        keep_tokens.append(token_dict[cur_token])
            token_dict = new_token_dict
            logging.info("vocab size after simplified: {}".format(len(keep_tokens)))
        else:
            keep_tokens = list(range(len(token_dict)))

        return token_dict, keep_tokens

    def __init__(self, token_dict):
        """初始化
        """
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'

        self.token2id = token_dict
        self.id2token = {v: k for k, v in token_dict.items()}

        # 给出各符号的id
        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            try:
                _token_id = token_dict[getattr(self, "_token_"+str(token))]
                logging.info("{} id: {}".format(token.upper(), _token_id))
                setattr(self, "_token_"+str(token)+"_id", _token_id)
            except Exception as e :
                logging.warning("no {} in vocab".format(token.upper()))

        self.vocab_size = len(token_dict)

    def save(self, save_path, encoding="utf-8"):
        def gen_vocab_list():
            for vocab, _ in sorted(self.token2id.items(), key=lambda x: x[1]):
                yield vocab
        write_to_file(gen_vocab_list(), save_path, encoding=encoding)

    def tokenize(self, text: str, add_cls=True, add_sep=True, max_length=None):
        """将text切分为token序列 并截断到最大长度max_length
        """
        tokens = self._tokenize(text)
        if add_cls:
            tokens.insert(0, self._token_cls)

        if add_sep:
            tokens.append(self._token_sep)

        if max_length is not None:
            # pop的时候 pop倒数第二个 因为倒数第一个是sep符号
            self.truncate_sequence(max_length, tokens, None, -2)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(self,
                          max_length,
                          first_sequence: List[str],
                          second_sequence=None,
                          pop_index=-1):
        """截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            # 不断截断 直到拼接长度小于max_length
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """

        first_tokens = self.tokenize(first_text)

        # 先处理第二个句子
        if second_text is None:
            second_tokens = None
        else:
            # 第二个句子 没有cls开始符号
            second_tokens = self.tokenize(second_text, add_cls=False)

        if max_length is not None:
            # 两句话长度和截断到max_length
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        # 如果有指定限制第一句话的长度 则第一句话在指定长度内
        # 若第一句话没有指定长度 其余pad
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                    (first_length - len(first_token_ids)))
        # 第一句话segment_ids为全0
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            # 如果有指定限制第二句话的长度 则第二句话在指定长度内
            # 若第二句话没有指定长度 其余pad
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend([self._token_pad_id] *
                        (second_length - len(second_token_ids)))
            # 第二句话segment_ids为全1
            second_segment_ids = [1] * len(second_token_ids)

            # 有第二句话的话 将其加在第一句话之后
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) + " " for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class BertTokenizer(BasicTokenizer):

    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self.token2id.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        return self.id2token[i]

    def decode(self, ids=None, tokens=None):
        """转为可读文本
        """
        assert ids is not None or tokens is not None,\
                "either ids or tokens is required"
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                # 将subword合并
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                # cjk文件直接加
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                # cjk文件直接加
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self, text):
        """基本分词函数
        """
        # 转小写 
        text = text.lower()
        # 特殊组合字符分为多个字符表示
        text = unicodedata.normalize('NFD', text)
        # 去除mark nonspacing类符号 例如音标等
        text = ''.join([
            ch for ch in text if unicodedata.category(ch) != 'Mn'])
        spaced = ''
        for ch in text:
            # 标点符号 和 CJK 类字符 按单字符切分
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            # 空白类字符统一用一个空格表示
            elif self._is_space(ch):
                spaced += ' '
            # 控制类字符忽略
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            # 其余的原样加
            else:
                spaced += ch

        tokens = []
        # 按空格切分
        # 因此 标点符号、单字 均被单独切分
        # 尝试将英文单词切分为subword
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        # 如果word在token2id中 直接返回
        if word in self.token2id:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.token2id:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens
