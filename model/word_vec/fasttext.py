#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   fasttext.py
Author:   zhanghao55@baidu.com
Date  :   20/07/25 15:47:29
Desc  :   
"""

from gensim.models.fasttext import FastText as ft
import logging
import os
import time


def load_fasttext_model(model_path):
    ft_model = None
    start_time = time.time()
    if os.path.exists(model_path):
        logging.info("model load from: %s" % model_path)
        ft_model = ft.load(model_path)
    else:
        logging.error("model_path doesn't exist.")
    logging.info("cost time = %.4fs" % (time.time() - start_time))
    return ft_model


def fasttext_training(data_iter_gen_func, model_save_path=None, emb_size=128, epochs=5, window=3, min_count=3, workers=20):
    """
    """
    start_time = time.time()
    ft_model = None
    if model_save_path is not None:
        ft_model = load_fasttext_model(model_save_path)

    vocab_update = True
    # 如果没有储存的模型 或者模型加载失败 则初始化模型
    if ft_model is None:
        logging.info("model initialized")
        ft_model = ft(size=emb_size, window=window, min_count=min_count, workers=workers)
        vocab_update = False

    # 将数据生成迭代器的函数再包装成迭代器
    class DataIter(object):
        def __iter__(self):
            return data_iter_gen_func()

    logging.info("build vocab start")
    start_time = time.time()
    ft_model.build_vocab(sentences=DataIter(), update=vocab_update)
    logging.info("cost time = %.4fs" % (time.time() - start_time))

    total_examples = ft_model.corpus_count
    logging.info("total examples = {}".format(total_examples))
    start_time = time.time()
    logging.info("model train start")
    ft_model.train(sentences=DataIter(), total_examples=total_examples, epochs=epochs)
    logging.info("cost time = %.4fs" % (time.time() - start_time))

    if model_save_path is not None:
        logging.info("model save start")
        start_time = time.time()
        ft_model.save(model_save_path)
        logging.info("cost time = %.4fs" % (time.time() - start_time))

    return ft_model


if __name__ == "__main__":
    pass
