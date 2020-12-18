#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: softmax.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/12/11 10:29:24
"""
import numpy as np

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D or list
    if isinstance(X, list) or len(X.shape) == 1:
        p = p.flatten()

    return p


if __name__ == "__main__":
    X = np.array([
            [1.1, 5.0, 2.2, 7.3],
            [6.5, 3.2, 8.8, 5.3],
            [2.7, 5.1, 9.6, 7.4],
        ])

    print("test softmax on:")
    print(X)
    print("="*30)

    print("softmax over rows")
    print(softmax(X, theta = 0.5, axis = 0))

    print("softmax over columns")
    print(softmax(X, theta = 0.5, axis = 1))

    print("softmax over columns, and squash it!")
    print(softmax(X, theta = 500.0, axis = 1))

    X = [1.1, 5.0, 2.2, 7.3]
    print("test softmax on:")
    print("="*30)

    print("softmax over columns")
    print(softmax(X, theta = 0.5, axis = 1))
