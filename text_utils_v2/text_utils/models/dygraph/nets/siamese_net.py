#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   siamese_net.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 17:46:03
Desc  :   
"""

import paddle.fluid as F
import paddle.fluid.layers as L

from ernie.modeling_ernie import ErnieModel
from ernie.file_utils import add_docstring


class SiameseNet(ErnieModel):
    """孪生网络
    """
    def __init__(self, cfg, name=None):
        super(SiameseNet, self).__init__(cfg, name=name)

        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        self.classifier = _build_linear(cfg['hidden_size'], cfg['num_labels'], append_name(name, 'cls'), initializer)

        prob = cfg.get('classifier_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    #@add_docstring(ErnieModel.forward.__doc__)
    #def forward(self, *args, **kwargs):
    #    """
    #    Args:
    #        labels (optional, `Variable` of shape [batch_size]): 
    #            ground truth label id for each sentence
    #    Returns:
    #        loss (`Variable` of shape []):
    #            Cross entropy loss mean over batch
    #            if labels not set, returns None
    #        logits (`Variable` of shape [batch_size, hidden_size]):
    #            output logits of classifier
    #    """
    #    labels = kwargs.pop('labels', None)
    #    pooled, encoded = super(ErnieModelForSequenceClassification, self).forward(*args, **kwargs)
    #    hidden = self.dropout(pooled)
    #    logits = self.classifier(hidden)

    #    if labels is not None:
    #        if len(labels.shape) == 1:
    #            labels = L.reshape(labels, [-1, 1])
    #        loss = L.softmax_with_cross_entropy(logits, labels)
    #        loss = L.reduce_mean(loss)
    #    else:
    #        loss = None
    #    return loss, logits

    def forward_once(self, src_ids):
        return super(ErnieModelForSequenceClassification, self).forward(src_ids)

    @add_docstring(ErnieModel.forward.__doc__)
    def forward(self, input1, input2, *args, **kwargs):
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


