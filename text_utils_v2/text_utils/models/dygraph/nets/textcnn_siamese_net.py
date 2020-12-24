#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   textcnn_siamese_net.py
Author:   zhanghao55@baidu.com
Date  :   20/12/23 19:49:31
Desc  :   
"""

import logging
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from text_utils.models.dygraph.nets.siamese_net import SiameseNet
from text_utils.models.dygraph.nets.textcnn import TextCNN


class TextCNNSiameseNet(SiameseNet):
    """孪生网络
    """
    def __init__(self, vocab_size, emb_dim=32, is_sparse=True, hidden_dropout_prob=0.1, triplet_margin=1.0, *args, **kwargs):
        super(TextCNNSiameseNet, self).__init__()
        self.triplet_margin = triplet_margin
        logging.info("triplet_margin: {}".format(triplet_margin))

        self.embedding = D.Embedding(
            size=[vocab_size, emb_dim],
            dtype='float32',
            is_sparse=is_sparse,
            )

        self.textcnn = TextCNN(emb_dim, *args, **kwargs)

        logging.info("feature dropout: {}".format(hidden_dropout_prob))

        self.dropout = lambda i: L.dropout(i,
                dropout_prob=hidden_dropout_prob,
                dropout_implementation="upscale_in_train",
                ) if self.training else i

    def _forward_once(self, inputs):
        emb = self.embedding(inputs)
        conv_pool_res = self.textcnn(emb)
        return conv_pool_res

    def forward(self, anchor_src, pos_src=None, neg_src=None):
        """
        Args:
            logits_softmax (optional, boolean):
                if true, return logits after softmax
        Returns:
            loss (`Variable` of shape []):
                Cross entropy loss mean over batch
                if labels not set, doesn't return
            logits (`Variable` of shape [batch_size, hidden_size]):
                output logits of classifier
        """
        anchor_emb = self._forward_once(anchor_src)
        anchor_emb_dropout = self.dropout(anchor_emb)

        if pos_src is None:
            return anchor_emb_dropout

        pos_emb = self._forward_once(pos_src)
        pos_emb_dropout = self.dropout(pos_emb)

        #logging.info("anchor emb shape: {}".format(anchor_emb.shape))
        #logging.info("pos emb shape: {}".format(pos_emb.shape))
        #logging.info("anchor emb dropoutshape: {}".format(anchor_emb_dropout.shape))
        #logging.info("pos emb dropout shape: {}".format(pos_emb_dropout.shape))

        distance_pos = self._distance(anchor_emb, pos_emb)

        # 确认当前模式
        if neg_src is None:
            # 当neg_src为None时 说明只需要得到anchor_src和pos_src的距离
            return distance_pos, anchor_emb_dropout, pos_emb_dropout

        # 否则 需要得到anchor_src和neg_src的距离 并计算两距离的triplet loss
        neg_emb = self._forward_once(neg_src)
        neg_emb_dropout = self.dropout(neg_emb)
        distance_neg = self._distance(anchor_emb, neg_emb)
        triplet_loss = self._triplet_loss(distance_pos, distance_neg, self.triplet_margin)
        return triplet_loss, distance_pos, distance_neg


if __name__ == "__main__":
    pass


