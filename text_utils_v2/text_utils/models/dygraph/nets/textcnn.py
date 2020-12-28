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


class ConvPool(D.Layer):
    """卷积池化层
    """
    def __init__(self,
            num_channels,
            num_filters,
            filter_size,
            padding,
            use_cudnn=False,
            ):
        super(ConvPool, self).__init__()

        self._conv2d = D.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            use_cudnn=use_cudnn,
            act='tanh')

    def forward(self, inputs):
        """前向预测
        """
        # inputs shape = [batch_size, num_channels, seq_len, emb_dim] [N, C, H, W]
        #print("inputs shape: {}".format(inputs.shape))

        # x shape = [batch_size, num_filters, height_after_conv, width_after_conv=1]
        x = self._conv2d(inputs)
        #print("conv3d shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters, height_after_pool=1, width_after_pool=1]
        x = L.reduce_max(x, dim=2, keep_dim=True)
        #print("reduce sum shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters]
        x = L.squeeze(x, axes=[2, 3])
        return x


class TextCNN(D.Layer):
    """textcnn层
    """
    def __init__(self,
            emb_dim,
            num_filters=10,
            num_channels=1,
            win_size_list=None,
            use_cudnn=True,
            ):
        super(TextCNN, self).__init__()

        #logging.info("emb_dim       = {}".format(emb_dim))
        #logging.info("num filters   = {}".format(num_filters))
        #logging.info("num channels  = {}".format(num_channels))
        #logging.info("win size list = {}".format(win_size_list))
        #logging.info("is sparse     = {}".format(is_sparse))
        #logging.info("use cudnn     = {}".format(use_cudnn))

        if win_size_list is None:
            win_size_list = [3]

        def gen_conv_pool(win_size):
            """生成指定窗口的卷积池化层
            """
            return ConvPool(
                    num_channels,
                    num_filters,
                    [win_size, emb_dim],
                    padding=[1, 0],
                    use_cudnn=use_cudnn,
                    )

        self.conv_pool_list = D.LayerList([gen_conv_pool(win_size) for win_size in win_size_list])


    def forward(self, input_emb):
        """前向预测
        """
        # emb shape = [batch_size, 1, seq_len, emb_dim]
        emb = L.unsqueeze(input_emb, axes=[1])
        #print("emb shape: {}".format(emb.shape))

        conv_pool_res_list = [conv_pool(emb) for conv_pool in self.conv_pool_list]

        conv_pool_res = L.concat(conv_pool_res_list, axis=-1)

        return conv_pool_res


class TextCNNClassifier(D.Layer):
    """textcnn分类模型
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

        self.embedding = D.Embedding(
            size=[vocab_size, emb_dim],
            dtype='float32',
            is_sparse=is_sparse,
            )

        self.textcnn = TextCNN(
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
        """前向预测
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

        # 输出logits为softmax后的结果
        if logits_softmax:
            logits = L.softmax(logits)

        # 如果没有给标签 则输出logits结果
        if labels is None:
            return logits

        # 调整label的形状
        if len(labels.shape) == 1:
            labels = L.reshape(labels, [-1, 1])
        #logging.info("labels shape: {}".format(labels.shape))

        loss = L.softmax_with_cross_entropy(logits, labels)
        # 如果输出logits的激活函数为softmax 则不能用softmax_with_cross_entropy
        #loss = L.cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
        #acc = L.accuracy(input=prediction, label=label)
        return loss, logits
