#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   mark_data_clean.py
Author:   zhanghao55@baidu.com
Date  :   20/09/24 15:02:29
Desc  :   
"""

import sys
from cleanlab.pruning import get_noise_indices


def wrong_label_detect(pred_prob, given_label)
    wrong_label_indexs = get_noise_indices(
            s=given_label,
            psx=pred_prob,
            sorted_index_method='normalized_margin', # Orders label errors
            )
    return wrong_label_indexs


if __name__ == "__main__":
    pass


