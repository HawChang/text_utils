#!/usr/bin/env python
# -*- coding: gb18030 -*-
 
"""
Author:   zhanghao55@baidu.com
Date  :   19/09/19 21:12:03
DESC  :   ͨ���д���
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
    """ͨ���д���
    """
    def __init__(self,
            seg_method="word_seg",
            segdict_path="./dict/chinese_gbk",
            jieba_tmp_dir=None):
        """�д����ʼ��
        [in] seg_method: str, ָ���дʷ�ʽ jieba �����ڲ��дʹ���
             segdict_path: str, �ڲ��дʹ�����Ҫ�д��ֵ��ַ
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
        """ʹ�ù�˾�ڲ��дʹ���
        [in]  text: str, ���д��ַ����� unicode��gb18030����
        [out] seg_list: list[str], �дʽ����unicode����
        """
        if six.PY2 and isinstance(text, unicode):
            text = text.encode("gb18030", "ignore")
        # �ڲ��дʺ�������gbk�����ַ���
        return [x.decode("gb18030") for x in self._segger.seg_words(text)]

    def jieba_seg_words(self, text):
        """ʹ�ý�ͽ��зִ�
        [in]  text: str, ���д��ַ�����unicode��gb18030����
        [out] seg_list: list[str], �дʽ����unicode����
        """
        if six.PY2 and isinstance(text, unicode):
                text = text.encode("gb18030", "ignore")
        # jieba�ִʽ����unicode����
        return [x for x in jieba.lcut(text)]

    def destroy(self):
        """�ڲ��дʹ�����Ҫ�ͷ��ڴ�
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
    ## ����
    #segger = WordSegger(segdict_path="./dict/chinese_gbk")
    ##segger = WordSegger(seg_method="jieba")
    #print(" ".join(segger.seg_words("���Ը��д��ദ��gb18030������ַ���")))
    #print(" ".join(segger.seg_words(u"�ٿ���unicode������ַ����Ƿ�Ҳ����")))
    #print(" ".join(segger.seg_words(u"���ӽ��ӣ��������ٶȣ�ȥ�����۾���")))
    #print(" ".join(segger.seg_words(u"�ǻ�ʧ�ܺ󱻼�ʥּ������ʷ����ߵ�̫����˭��")))
    #segger.destroy()
    stream_seg(sys.stdin)
