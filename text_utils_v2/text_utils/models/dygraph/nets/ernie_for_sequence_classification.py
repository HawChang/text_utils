#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   ernie_for_sequence_classification.py
Author:   zhanghao55@baidu.com
Date  :   20/09/19 17:07:45
Desc  :   ��lossΪNoneʱ��ֻ���logits���������Զ��������ṹƥ��
"""

import paddle.fluid.layers as L
from ernie.modeling_ernie import ErnieModel
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie.file_utils import add_docstring


class ErnieSequenceClassificationCustomized(ErnieModelForSequenceClassification):
    """���ƻ�ernie����ģ�� 
        1. forward�������logits_softmax����
        2. û��labelʱ, ֻ����logits
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
        loss, logits = super(ErnieSequenceClassificationCustomized, self).forward(*args,
                **kwargs)
        if logits_softmax:
            logits = L.softmax(logits, use_cudnn=True)

        if loss is None:
            return logits
        else:
            return loss, logits


if __name__ == "__main__":
    pass


