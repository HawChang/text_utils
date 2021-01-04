#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   gru.py
Author:   zhanghao55@baidu.com
Date  :   20/08/26 16:27:01
Desc  :   
"""

import logging
import numpy as np
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from basic_layers import DynamicGRULayer, EmbeddingLayer


class GRUClassifier(D.Layer):
    """GRU分类模型
    """
    def __init__(self,
            num_class,
            vocab_size,
            emb_dim=128,
            gru_dim=256,
            fc_hid_dim=256,
            is_sparse=True,
            bi_direction=True,
            ):
        super(GRUClassifier, self).__init__()

        logging.info("num_class    = {}".format(num_class))
        logging.info("vocab_size   = {}".format(vocab_size))
        logging.info("emb_dim      = {}".format(emb_dim))
        logging.info("gru_dim      = {}".format(gru_dim))
        logging.info("fc_hid_dim   = {}".format(fc_hid_dim))
        logging.info("is_sparse    = {}".format(is_sparse))
        logging.info("bi_direction = {}".format(bi_direction))

        self.bi_direction = bi_direction

        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dtype='float32',
            is_sparse=is_sparse)

        self._hid_fc1 = D.Linear(input_dim=emb_dim, output_dim=gru_dim * 3)

        self._gru_forward = DynamicGRULayer(size=gru_dim, h_0=None, is_reverse=False)

        if bi_direction:
            self._gru_backward = DynamicGRULayer(size=gru_dim, h_0=None, is_reverse=True)
            self._hid_fc2 = D.Linear(input_dim=gru_dim * 2, output_dim=fc_hid_dim, act="tanh")
        else:
            self._hid_fc2 = D.Linear(input_dim=gru_dim, output_dim=fc_hid_dim, act="tanh")

        self._output_fc = D.Linear(input_dim=fc_hid_dim, output_dim=num_class, act=None)

    def forward(self, inputs, labels=None, logits_softmax=False):
        """前向预测
        """
        emb = self.embedding(inputs)

        hid_fc1 = self._hid_fc1(emb)

        gru_forward = self._gru_forward(hid_fc1)
        gru_forward_tanh = L.tanh(gru_forward)
        if self.bi_direction:
            gru_backward = self._gru_backward(hid_fc1)
            gru_backward_tanh = L.tanh(gru_backward)
            encoded_vector = L.concat(
                input=[gru_forward_tanh, gru_backward_tanh], axis=2)
            encoded_vector = L.reduce_max(encoded_vector, dim=1)
        else:
            encoded_vector = L.reduce_max(gru_forward_tanh, dim=1)

        hid_fc_2 = self._hid_fc2(encoded_vector)

        logits = self._output_fc(hid_fc_2)

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
