#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   textcnn.py
Author:   zhanghao55@baidu.com
Date  :   20/08/11 11:43:17
Desc  :   
"""

import logging
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from basic_layers import TextCNNLayer, EmbeddingLayer


class TextCNNClassifier(D.Layer):
    """textcnn����ģ��
    """
    def __init__(self,
            num_class,
            vocab_size,
            emb_dim=32,
            num_filters=10,
            fc_hid_dim=32,
            num_channels=1,
            win_size_list=None,
            is_sparse=True,
            use_cudnn=True,
            ):
        super(TextCNNClassifier, self).__init__()

        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dtype='float32',
            is_sparse=is_sparse,
            )

        self.textcnn = TextCNNLayer(
            emb_dim,
            num_filters,
            num_channels,
            win_size_list,
            use_cudnn,
            )

        logging.info("num_class     = {}".format(num_class))
        logging.info("vocab size    = {}".format(vocab_size))
        logging.info("emb_dim       = {}".format(emb_dim))
        logging.info("num filters   = {}".format(num_filters))
        logging.info("fc_hid_dim    = {}".format(fc_hid_dim))
        logging.info("num channels  = {}".format(num_channels))
        logging.info("win size list = {}".format(win_size_list))
        logging.info("is sparse     = {}".format(is_sparse))
        logging.info("use cudnn     = {}".format(use_cudnn))

        self._hid_fc = D.Linear(input_dim=num_filters * len(win_size_list), output_dim=fc_hid_dim, act="tanh")
        self._output_fc = D.Linear(input_dim=fc_hid_dim, output_dim=num_class, act=None)

    def forward(self, inputs, labels=None, logits_softmax=False):
        """ǰ��Ԥ��
        """
        #print("\n".join(map(lambda ids: "/ ".join([id_2_token[x] for x in ids]), inputs.numpy())))
        # inputs shape = [batch_size, seq_len]
        #print("inputs shape: {}".format(inputs.shape))

        # emb shape = [batch_size, seq_len, emb_dim]
        emb = self.embedding(inputs)
        #print("emb shape: {}".format(emb.shape))

        conv_pool_res = self.textcnn(emb)

        hid_fc = self._hid_fc(conv_pool_res)
        #print("hid_fc shape: {}".format(hid_fc.shape))

        logits = self._output_fc(hid_fc)
        #print("logits shape: {}".format(logits.shape))

        # ���logitsΪsoftmax��Ľ��
        if logits_softmax:
            logits = L.softmax(logits)

        # ���û�и���ǩ �����logits���
        if labels is None:
            return logits

        # ����label����״
        if len(labels.shape) == 1:
            labels = L.reshape(labels, [-1, 1])
        #logging.info("labels shape: {}".format(labels.shape))

        loss = L.softmax_with_cross_entropy(logits, labels)
        # ������logits�ļ����Ϊsoftmax ������softmax_with_cross_entropy
        #loss = L.cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
        #acc = L.accuracy(input=prediction, label=label)
        return loss, logits
