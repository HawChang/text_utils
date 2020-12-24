#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   siamese_net.py
Author:   zhanghao55@baidu.com
Date  :   20/12/23 19:32:48
Desc  :   
"""

import logging
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

class SiameseNet(D.Layer):
    """孪生网络
    """
    def __init__(self, *args, **kwargs):
        super(SiameseNet, self).__init__(*args, **kwargs)
        #self.triplet_margin = triplet_margin
        #logging.info("triplet_margin: {}".format(self.triplet_margin))

    def _distance(self, anchor_emb, other_emb):
        """计算两输出矩阵的距离
        """
        square_out = L.square(anchor_emb - other_emb)
        #logging.info("square_out shape: {}".format(square_out.shape))
        distance = L.reduce_sum(square_out, 1)
        #logging.info("distance shape: {}".format(distance.shape))
        return distance

    def _triplet_loss(self, distance_pos, distance_neg, triplet_margin=1.0):
        """计算两距离的loss
        """
        distance = distance_pos - distance_neg
        #logging.info("distance: {}".format(distance))

        distance_margin = distance + triplet_margin
        #logging.info("distance margin: {}".format(distance_margin))

        loss = L.relu(distance_margin)
        #logging.info("loss: {}".format(loss))
        #logging.info("loss shape: {}".format(loss.shape))
        loss = L.reduce_mean(loss)
        #logging.info("loss reduce mean: {}".format(loss))
        #logging.info("loss reduce mean shape: {}".format(loss.shape))
        return loss


if __name__ == "__main__":
    pass


