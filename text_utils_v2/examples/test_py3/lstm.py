#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   lstm.py
Author:   zhanghao55@baidu.com
Date  :   20/12/29 15:12:05
Desc  :   
"""

import logging
import numpy as np
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from basic_layers import DynamicLSTMLayer, EmbeddingLayer


class DynamicLSTMClassifier(D.Layer):
    """LSTM分类模型
    """
    def __init__(self,
            num_class,
            vocab_size,
            emb_dim=128,
            lstm_dim=256,
            fc_hid_dim=256,
            is_sparse=True,
            bi_direction=True,
            dropout_prob=0.1,
            ):
        super(DynamicLSTMClassifier, self).__init__()

        logging.info("num_class    = {}".format(num_class))
        logging.info("vocab_size   = {}".format(vocab_size))
        logging.info("emb_dim      = {}".format(emb_dim))
        logging.info("lstm_dim      = {}".format(lstm_dim))
        logging.info("fc_hid_dim   = {}".format(fc_hid_dim))
        logging.info("is_sparse    = {}".format(is_sparse))
        logging.info("bi_direction = {}".format(bi_direction))
        logging.info("dropout_prob = {}".format(dropout_prob))

        self.bi_direction = bi_direction

        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dtype='float32',
            is_sparse=is_sparse)

        self._lstm_forward = DynamicLSTMLayer(input_size=emb_dim, hidden_size=lstm_dim, is_reverse=False)

        if bi_direction:
            self._lstm_backward = DynamicLSTMLayer(input_size=emb_dim, hidden_size=lstm_dim, is_reverse=True)
            self._hid_fc2 = D.Linear(input_dim=lstm_dim * 2, output_dim=fc_hid_dim, act="tanh")
        else:
            self._hid_fc2 = D.Linear(input_dim=lstm_dim, output_dim=fc_hid_dim, act="tanh")

        self._output_fc = D.Linear(input_dim=fc_hid_dim, output_dim=num_class, act=None)

        self.dropout = lambda i: L.dropout(i,
                dropout_prob=dropout_prob,
                dropout_implementation="upscale_in_train") if self.training else i

    def forward(self, inputs, labels=None, logits_softmax=False):
        """前向预测
        """
        #logging.info("inputs shape: {}".format(inputs.shape))
        emb = self.embedding(inputs)
        #logging.info("emb shape: {}".format(emb.shape))

        emb_dropout = self.dropout(emb)

        lstm_forward, _ = self._lstm_forward(emb_dropout)
        #logging.info("lstm_forward shape: {}".format(lstm_forward.shape))
        lstm_forward_tanh = L.tanh(lstm_forward)
        if self.bi_direction:
            lstm_backward, _ = self._lstm_backward(emb_dropout)
            lstm_backward_tanh = L.tanh(lstm_backward)
            encoded_vector = L.concat(
                input=[lstm_forward_tanh, lstm_backward_tanh], axis=-1)
            encoded_vector = L.reduce_max(encoded_vector, dim=1)
        else:
            encoded_vector = L.reduce_max(lstm_forward_tanh, dim=1)

        #logging.info("encoded_vector shape: {}".format(encoded_vector.shape))

        hid_fc_2 = self._hid_fc2(encoded_vector)
        #logging.info("hid_fc_2 shape: {}".format(hid_fc_2.shape))

        logits = self._output_fc(hid_fc_2)
        #logging.info("logits shape: {}".format(logits.shape))

        # 输出logits为softmax后的结果
        if logits_softmax:
            logits = L.softmax(logits)

        # 如果没有给标签 则输出logits结果
        if labels is None:
            return logits

        if len(labels.shape) == 1:
            labels = L.reshape(labels, [-1, 1])
        #print("labels shape: {}".format(labels.shape))

        loss = L.softmax_with_cross_entropy(logits, labels)
        # 如果输出logits的激活函数为softmax 则不能用softmax_with_cross_entropy
        #loss = L.cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
        return loss, logits
