#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   ernie_siamese_net.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 17:46:03
Desc  :   
"""

import logging
import paddle.fluid.layers as L

from text_utils.models.dygraph.nets.siamese_net import SiameseNet
from ernie.modeling_ernie import ErnieModel
from ernie.file_utils import add_docstring


class ErnieSiameseNet(ErnieModel, SiameseNet):
    """孪生网络
    """
    def __init__(self, cfg, name=None):
        super(ErnieSiameseNet, self).__init__(cfg, name=None)

        self.triplet_margin = cfg.pop("triplet_margin", 1.0)
        logging.info("triplet_margin: {}".format(self.triplet_margin))

        prob = cfg.get('classifier_dropout_prob', cfg['hidden_dropout_prob'])
        logging.info("emb dropout: {}".format(prob))

        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    def _forward_once(self, src_ids, *args, **kwargs):
        pooled, encoded = ErnieModel.forward(self, src_ids, *args, **kwargs)
        return pooled

    @add_docstring(ErnieModel.forward.__doc__)
    def forward(self, anchor_src, pos_src=None, neg_src=None, **kwargs):
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
        anchor_emb = self._forward_once(anchor_src, **kwargs)
        anchor_emb_dropout = self.dropout(anchor_emb)

        if pos_src is None:
            return anchor_emb_dropout

        pos_emb = self._forward_once(pos_src,  **kwargs)
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
        neg_emb = self._forward_once(neg_src, **kwargs)
        neg_emb_dropout = self.dropout(neg_emb)
        distance_neg = self._distance(anchor_emb, neg_emb)
        triplet_loss = self._triplet_loss(distance_pos, distance_neg, self.triplet_margin)
        return triplet_loss, distance_pos, distance_neg


if __name__ == "__main__":
    pass


