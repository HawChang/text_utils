#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: feature_vectorizer.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/20 21:04:00
"""

import sys
reload(sys)
sys.setdefaultencoding("gb18030")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def init_vectorizer(vec_method="count", lowercase=True, min_df=2, token_pattern=r'(?u)[^ ]+', **vec_params):
    """向量化
    """
    return {"count": CountVectorizer,
            "tfidf": TfidfVectorizer}[vec_method](token_pattern=token_pattern, lowercase=lowercase, min_df=min_df, **vec_params)
