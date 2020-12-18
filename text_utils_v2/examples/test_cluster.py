#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_ernie_feature_extract.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 11:15:48
Desc  :   
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import logging
import paddle.fluid.dygraph as D
#import time
import unittest

from ernie.modeling_ernie import ErnieModel

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.tokenizers.lr_tokenizer import LRTokenizer
from text_utils.utils.data_io import get_attr_values, gen_batch_data
from text_utils.utils.data_io import write_to_file
from text_utils.models.dygraph.train_infer_utils import batch_infer
from text_utils.utils.logger import init_log
from text_utils.models.machine_learning.cluster import mini_batch_kmeans, data_cluster
from text_utils.utils.vectorizer import init_vectorizer

init_log()

class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestCluster.test_output_dir = os.path.join(test_root, "output/test_clutser/")
        if not os.path.isdir(TestCluster.test_output_dir):
            os.mkdir(TestCluster.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/classification_data/toutiao_news")

        example_num = 5

        # 加载数据
        data_path = os.path.join(test_data_dir, "toutiao_news_shrink.txt")
        TestCluster.text_list, TestCluster.keywords_list = \
                get_attr_values(data_path, fetch_list=["text", "keywords"], encoding="utf-8")
        logging.info("data num = {}".format(len(TestCluster.text_list)))

        TestCluster.text_list = TestCluster.text_list
        TestCluster.keywords_list = TestCluster.keywords_list

        logging.info(u"数据样例")
        for index, text in enumerate(TestCluster.text_list[:example_num]):
            logging.info("example #{}:".format(index))
            logging.info("text: {}".format(text.encode("utf-8")))

        TestCluster.cluster_num = 15
        max_no_improvement = 10000

        TestCluster.cluster_model = mini_batch_kmeans(n_clusters=TestCluster.cluster_num, max_no_improvement=max_no_improvement)

    def test_ernie_cluster(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                }

        tokenizer = ErnieTokenizer.load("./dict/vocab.txt")
        text_ids = tokenizer.transform(TestCluster.text_list)

        with D.guard():
            ernie = ErnieModel.from_pretrained(**ernie_config)
            res = batch_infer(ernie, text_ids, batch_size=128, with_label=False, logits_softmax=None)

        pooled_encode_vec, _ = zip(*res)
        data_cluster(TestCluster.cluster_model, pooled_encode_vec)

        cluster_ids = TestCluster.cluster_model.labels_
        cluster_res_path = os.path.join(TestCluster.test_output_dir, "ernie_kmeans_cluster_%d.txt" % TestCluster.cluster_num)
        write_to_file(zip(cluster_ids, TestCluster.text_list), cluster_res_path, write_func=lambda x: "%s\t%s" % x)

    def test_ngram_count_cluster(self):
        tokenizer = LRTokenizer(
                stopword_path="./dict/stopword_shrink.txt",
                jieba_tmp_dir="./dict/jieba_tmp",
                )
        features_list = tokenizer.transform(TestCluster.text_list)
        features_list = [" ".join(x) for x in features_list]
        vectorizer = init_vectorizer(vec_method="count")
        feature_vec = vectorizer.fit_transform(features_list)
        logging.info("vocab size: {}".format(len(vectorizer.vocabulary_)))

        data_cluster(TestCluster.cluster_model, feature_vec)

        cluster_ids = TestCluster.cluster_model.labels_
        cluster_res_path = os.path.join(TestCluster.test_output_dir, "ngram_count_kmeans_cluster_%d.txt" % TestCluster.cluster_num)
        write_to_file(zip(cluster_ids, TestCluster.text_list), cluster_res_path, write_func=lambda x: "%s\t%s" % x)

    def test_ngram_tfidf_cluster(self):
        tokenizer = LRTokenizer(
                stopword_path="./dict/stopword_shrink.txt",
                jieba_tmp_dir="./dict/jieba_tmp",
                )
        features_list = tokenizer.transform(TestCluster.text_list)
        features_list = [" ".join(x) for x in features_list]
        vectorizer = init_vectorizer(vec_method="tfidf")
        feature_vec = vectorizer.fit_transform(features_list)
        logging.info("vocab size: {}".format(len(vectorizer.vocabulary_)))

        data_cluster(TestCluster.cluster_model, feature_vec)

        cluster_ids = TestCluster.cluster_model.labels_
        cluster_res_path = os.path.join(TestCluster.test_output_dir, "ngram_tfidf_kmeans_cluster_%d.txt" % TestCluster.cluster_num)
        write_to_file(zip(cluster_ids, TestCluster.text_list), cluster_res_path, write_func=lambda x: "%s\t%s" % x)

if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    suit.addTest(TestCluster("test_ernie_cluster"))
    suit.addTest(TestCluster("test_ngram_count_cluster"))
    suit.addTest(TestCluster("test_ngram_tfidf_cluster"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

