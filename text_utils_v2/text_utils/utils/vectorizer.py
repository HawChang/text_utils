#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   vectorizer.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 17:08:37
Desc  :   
"""

import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def init_vectorizer(vec_method="count", lowercase=True, min_df=2, token_pattern=r'(?u)[^ ]+', **vec_params):
    """向量化
    """
    return {"count": CountVectorizer,
            "tfidf": TfidfVectorizer}[vec_method](token_pattern=token_pattern, lowercase=lowercase, min_df=min_df, **vec_params)
