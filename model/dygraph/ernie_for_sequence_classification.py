#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   ernie_for_sequence_classification.py
Author:   zhanghao55@baidu.com
Date  :   20/09/19 17:07:45
Desc  :   
"""

import sys

import paddle.fluid.layers as L
from ernie.modeling_ernie import ErnieModel
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie.file_utils import add_docstring


class ErnieModelCustomized(ErnieModelForSequenceClassification):
    """定制化ernie分类模型 
        1. forward函数添加logits_softmax参数
        2. 没有label时, 只返回logits
    """
    @add_docstring(ErnieModel.forward.__doc__)
    def forward(self, *args, **kwargs):
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
        logits_softmax = kwargs.pop("logits_softmax", False)
        loss, logits = super(ErnieModelCustomized, self).forward(*args,
                **kwargs)
        if logits_softmax:
            logits = L.softmax(logits, use_cudnn=True)

        if loss is None:
            return logits
        else:
            return loss, logits


if __name__ == "__main__":
    pass


