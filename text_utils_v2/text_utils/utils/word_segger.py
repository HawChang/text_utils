#!/usr/bin/env python
# -*- coding: gb18030 -*-
 
"""
Author:   zhanghao55@baidu.com
Date  :   19/09/19 21:12:03
DESC  :   通用切词类
"""

import jieba
import logging
import os
import six
try:
    import word_seg
    _word_seg_available = True
except ImportError as e:
    _word_seg_available = False
    logging.warning("import word_seg fail.")
    logging.warning(e)


class WordSegger(object):
    """通用切词类
    """
    def __init__(self,
            seg_method="word_seg",
            segdict_path="./dict/chinese_gbk",
            jieba_tmp_dir=None):
        """切词类初始化
        [in] seg_method: str, 指明切词方式 jieba 还是内部切词工具
             segdict_path: str, 内部切词工具需要切词字典地址
        """
        self.seg_words = {
            "jieba": self.jieba_seg_words,
            "word_seg": self.baidu_seg_words
        }[seg_method]

        self._segger = None
        if seg_method == "word_seg":
            if not _word_seg_available:
                raise RuntimeError("module word_seg is required when seg by word_seg")
            self._segger = word_seg.WordSeg(segdict_path)
            self._segger.init_wordseg_handle()
        elif seg_method == "jieba":
            if jieba_tmp_dir is not None:
                if os.path.isdir(jieba_tmp_dir):
                    jieba.dt.tmp_dir = jieba_tmp_dir
                else:
                    os.mkdir(jieba_tmp_dir)
                    logging.error("creat tmp dir for jieba: {}".format(jieba_tmp_dir))

    def baidu_seg_words(self, text):
        """使用公司内部切词工具
        [in]  text: str, 待切词字符串， unicode或gb18030编码
        [out] seg_list: list[str], 切词结果，unicode编码
        """
        if six.PY2 and isinstance(text, unicode):
            text = text.encode("gb18030", "ignore")
        # 内部切词函数接受gbk编码字符串
        return [x.decode("gb18030") for x in self._segger.seg_words(text)]

    def jieba_seg_words(self, text):
        """使用结巴进行分词
        [in]  text: str, 待切词字符串，unicode或gb18030编码
        [out] seg_list: list[str], 切词结果，unicode编码
        """
        if six.PY2 and isinstance(text, unicode):
                text = text.encode("gb18030", "ignore")
        # jieba分词结果是unicode编码
        return [x for x in jieba.lcut(text)]

    def destroy(self):
        """内部切词工具需要释放内存
        """
        if self._segger is not None:
            self._segger.destroy_wordseg_handle()

def stream_seg(in_stream):
    segger = WordSegger(seg_method="word_seg", segdict_path="./dict/chinese_gbk")
    for line in in_stream:
        line = line.strip("\n").decode("gb18030")
        print(" ".join(segger.seg_words(line)).encode("gb18030"))
    segger.destroy()


if __name__ == "__main__":
    import sys
    ## 测试
    #segger = WordSegger(segdict_path="./dict/chinese_gbk")
    ##segger = WordSegger(seg_method="jieba")
    #print(" ".join(segger.seg_words("测试该切词类处理gb18030编码的字符串")))
    #print(" ".join(segger.seg_words(u"再看看unicode编码的字符串是否也可以")))
    #print(" ".join(segger.seg_words(u"孩子近视，度数六百度，去哪配眼镜？")))
    #print(" ".join(segger.seg_words(u"登基失败后被假圣旨赐死，史上最悲催的太子是谁？")))
    #segger.destroy()
    stream_seg(sys.stdin)
