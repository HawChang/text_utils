#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: logger.py
Author: work(work@baidu.com)
Date: 2019/09/19 20:49:02
"""

import sys
sys.setdefaultencoding("gb18030")
reload(sys)

import logging


class Logger(object):
    _is_init = False	
    def __init__(self):
        if not self._is_init:
            logging.basicConfig(
                    #filename="log/run.log",
                    encoding="gb18030",
                    level=logging.DEBUG,
                    format="[%(asctime)s][%(filename)s:%(funcName)s:%(lineno)s][%(levelname)s]:%(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
            #ch = logging.StreamHandler()
            self.logger = logging.getLogger()
            #self.logger.addHandler(ch)
            self._is_init = True
    
    def get_logger(self):
        return self.logger

if __name__ == "__main__":
    pass
