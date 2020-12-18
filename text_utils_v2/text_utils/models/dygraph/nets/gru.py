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
#import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D


class DynamicGRU(D.Layer):
    """动态GRU层
    """
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False):
        super(DynamicGRU, self).__init__()
        self.gru_unit = D.GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        """前向预测
        """
        #print("inputs shape: {}".format(inputs.shape))
        # 初始化h_0
        if self.h_0 is None:
            batch_size, _, _ = inputs.shape
            self.h_0 = D.to_variable(
                    np.zeros((batch_size, self.size), dtype="float32"))

        hidden = self.h_0
        res = []
        input_trans = L.transpose(inputs, perm=[1, 0, 2])
        #print("input trans shape: {}".format(input_trans.shape))
        for i in range(input_trans.shape[0]):
            if self.is_reverse:
                i = input_trans.shape[0] - 1 - i
            cur_input = input_trans[i]

            #print("cur input shape: {}".format(cur_input.shape))
            hidden, reset, gate = self.gru_unit(cur_input, hidden)
            res.append(L.unsqueeze(input=hidden, axes=[1]))
        if self.is_reverse:
            res = res[::-1]
        res = L.concat(res, axis=1)
        #print("res shape: {}".format(res.shape))

        return res


class GRU(D.Layer):
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
        super(GRU, self).__init__()

        logging.info("num_class    = {}".format(num_class))
        logging.info("vocab_size   = {}".format(vocab_size))
        logging.info("emb_dim      = {}".format(emb_dim))
        logging.info("gru_dim      = {}".format(gru_dim))
        logging.info("fc_hid_dim   = {}".format(fc_hid_dim))
        logging.info("is_sparse    = {}".format(is_sparse))
        logging.info("bi_direction = {}".format(bi_direction))

        self.bi_direction = bi_direction

        self.embedding = D.Embedding(
            size=[vocab_size, emb_dim],
            dtype='float32',
            #param_attr=F.ParamAttr(learning_rate=30),
            is_sparse=is_sparse)

        self._hid_fc1 = D.Linear(input_dim=emb_dim, output_dim=gru_dim * 3)

        self._gru_forward = DynamicGRU(size=gru_dim, h_0=None, is_reverse=False)

        if bi_direction:
            self._gru_backward = DynamicGRU(size=gru_dim, h_0=None, is_reverse=True)
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
