#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   test_text_similarity.py
Author:   zhanghao55@baidu.com
Date  :   20/12/18 17:44:14
Desc  :   
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import codecs
import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as D
import random
import unittest

from collections import defaultdict
from itertools import combinations, permutations
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.utils.data_io import get_attr_values
from text_utils.utils.data_io import write_to_file
#from text_utils.utils.label_encoder import LabelEncoder
from text_utils.utils.logger import init_log
from text_utils.models.dygraph.nets.ernie_siamese_net import ErnieSiameseNet
from text_utils.models.dygraph.nets.textcnn_siamese_net import TextCNNSiameseNet
from text_utils.models.dygraph.base_model import SiameseModel
from text_utils.models.dygraph.train_infer_utils import train
from text_utils.utils.data_io import load_model

init_log()


class ErnieSiameseModel(SiameseModel):
    def build(self, **model_config):
        self.model = ErnieSiameseNet.from_pretrained(**model_config)
        self.built = True


class TextCNNSiameseModel(SiameseModel):
    def build(self, **model_config):
        self.model = TextCNNSiameseNet(**model_config)
        self.built = True


class TestTextSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_root = "./"
        TestTextSimilarity.test_output_dir = os.path.join(test_root, "output/test_text_similarity/")
        if not os.path.isdir(TestTextSimilarity.test_output_dir):
            os.mkdir(TestTextSimilarity.test_output_dir)

        test_data_dir = os.path.join(test_root, "dataset/text_similarity/")
        if not os.path.isdir(test_data_dir):
            os.mkdir(test_data_dir)

        pair_wise_data_path = os.path.join(test_data_dir, "pair_wise_data.txt")
        src_data_path = os.path.join(test_root, "dataset/classification_data/toutiao_news/toutiao_news_shrink.txt")

        if not os.path.exists(pair_wise_data_path):
            TestTextSimilarity.make_pairwise_data(src_data_path, pair_wise_data_path)

        src_text_list, src_label_list = \
                get_attr_values(src_data_path, fetch_list=["text", "label"], encoding="utf-8")

        src_data = zip(src_text_list, src_label_list)
        random.shuffle(src_data)
        TestTextSimilarity.src_text_list, TestTextSimilarity.src_label_list = zip(*src_data[:2000])

        anchor_list, pos_list, neg_list = \
                get_attr_values(pair_wise_data_path, fetch_list=["anchor", "pos", "neg"], encoding="utf-8")
        logging.info("data num = {}".format(len(anchor_list)))

        test_size = 0.2
        random_state = 1
        shuffle = True
        example_num = 5

        TestTextSimilarity.tokenizer = ErnieTokenizer.load("./dict/vocab.txt")

        TestTextSimilarity.src_text_ids = TestTextSimilarity.tokenizer.transform(TestTextSimilarity.src_text_list)

        #anchor_list = anchor_list[:500]
        #pos_list = pos_list[:500]
        #neg_list = neg_list[:500]

        anchor_ids = TestTextSimilarity.tokenizer.transform(anchor_list)
        pos_ids = TestTextSimilarity.tokenizer.transform(pos_list)
        neg_ids = TestTextSimilarity.tokenizer.transform(neg_list)

        text_list = zip(anchor_list, pos_list, neg_list)

        TestTextSimilarity.train_text, TestTextSimilarity.test_text, \
            train_anchor, test_anchor, train_pos, test_pos, train_neg, test_neg = \
            train_test_split(text_list, anchor_ids, pos_ids, neg_ids,
                             test_size=test_size, random_state=random_state, shuffle=shuffle)
        logging.info("train num = {}".format(len(train_anchor)))
        logging.info("test num = {}".format(len(test_anchor)))

        TestTextSimilarity.train_data = list(zip(train_anchor, train_pos, train_neg))
        TestTextSimilarity.eval_data = list(zip(test_anchor, test_pos, test_neg))

        logging.info(u"数据样例")
        for index, ((anchor_text, pos_text, neg_text), (anchor_ids, pos_ids, neg_ids)) in enumerate(zip(
                TestTextSimilarity.train_text[:example_num],
                TestTextSimilarity.train_data[:example_num],
                )):
            logging.info("example #{}:".format(index))
            logging.info("anchor_text: {}".format(anchor_text.encode("utf-8")))
            logging.info("anchor_ids: {}".format(anchor_ids))
            logging.info("anchor_ids type: {}".format(type(anchor_ids)))
            logging.info("anchor_ids dtype: {}".format(anchor_ids.dtype))
            logging.info("pos_text: {}".format(pos_text.encode("utf-8")))
            logging.info("pos_ids: {}".format(pos_ids))
            logging.info("pos_ids type: {}".format(type(pos_ids)))
            logging.info("pos_ids dtype: {}".format(pos_ids.dtype))
            logging.info("neg_text: {}".format(neg_text.encode("utf-8")))
            logging.info("neg_ids: {}".format(neg_ids))
            logging.info("neg_ids type: {}".format(type(neg_ids)))
            logging.info("neg_ids dtype: {}".format(neg_ids.dtype))

        logging.info(u"src数据样例")
        for index, (text_ids, text, label) in enumerate(zip(
            TestTextSimilarity.src_text_ids[:example_num],
            TestTextSimilarity.src_text_list[:example_num],
            TestTextSimilarity.src_label_list[:example_num],
            )):
            logging.info("example #{}:".format(index))
            logging.info("text: {}".format(text.encode("utf-8")))
            logging.info("label: {}".format(label.encode("utf-8")))
            logging.info("text_ids: {}".format(text_ids))

    @classmethod
    def make_pairwise_data(cls, src_data_path, output_path, num_each_label_pair=100):
        """
        """
        # 加载源数据
        text_list, label_list = \
                get_attr_values(src_data_path, fetch_list=["text", "label"], encoding="utf-8")
        logging.info("data num = {}".format(len(text_list)))

        # -------------------- 构建训练集 -------------------------
        label_text_dict = defaultdict(set)
        for text, label in zip(text_list, label_list):
            label_text_dict[label].add(text)

        label_text_dict = {k:list(v) for k, v in label_text_dict.items()}

        label_text_num_list = [(label, len(text_list)) for label, text_list in label_text_dict.items()]
        label_text_num_list = sorted(label_text_num_list,key=lambda x: x[1], reverse=True)
        logging.info(u"\n各类物料数:\n" + "\n".join(["%s = %d" % (label, text_num) for label, text_num in label_text_num_list]))

        def gen_pair_wise_data():
            # 头条物料有15类 每类样本量从340到4w左右
            # 建立pairwise训练物料
            for anchor_label, neg_label in permutations(label_text_dict.keys(), 2):
                # 每一类做anchor 其他的每一类做neg
                # neg类 随机抽num_each_label_pair个物料
                #neg_text_list = random.sample(label_text_dict[neg_label], min(num_each_label_pair, len(label_text_dict[neg_label])))

                # anchor类 随机抽num_each_label_pair个配对
                anchor_pair_list = list(combinations(label_text_dict[anchor_label], 2))
                random.shuffle(anchor_pair_list)
                for index, (anchor_text, pos_text) in enumerate(anchor_pair_list):
                    if index == num_each_label_pair:
                        break
                    neg_text = random.sample(label_text_dict[neg_label], 1)[0]

                    yield "\t".join([
                        anchor_text,
                        pos_text,
                        neg_text,
                        anchor_label,
                        neg_label,
                        ])

        with codecs.open(output_path, "w", "utf-8") as wf:
            wf.write("\t".join([
                "anchor",
                "pos",
                "neg",
                "pos_label",
                "neg_laebl",
                ]) + "\n")

            for text in gen_pair_wise_data():
                wf.write(text + "\n")

    def test_ernie_siamese_train(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                "triplet_margin": 1.1,
                }

        run_config = {
                "model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "ernie"),
                "best_model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "ernie_best"),
                "epochs": 2,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                "learning_rate": 5e-5,
                "load_best_model": False,
                }

        with D.guard():
            ernie_siamese = ErnieSiameseModel()
            ernie_siamese.build(**ernie_config)
            best_acc = ernie_siamese.train(
                    TestTextSimilarity.train_data, TestTextSimilarity.eval_data,
                    **run_config)
        logging.info("ernie siamese est train score: {}".format(best_acc))

    def test_ernie_siamese_infer(self):
        ernie_config = {
                "pretrain_dir_or_url": "ernie-1.0",
                "triplet_margin": 1.1,
                }

        infer_config = {
                "best_model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "ernie_best"),
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                }

        topk=10

        infer_res_path = os.path.join(TestTextSimilarity.test_output_dir, "ernie_infer_res.txt")

        with D.guard():
            ernie_siamese = ErnieSiameseModel()
            ernie_siamese.build(**ernie_config)
            ernie_siamese.load_model(infer_config["best_model_save_path"])
            text_emb_list = ernie_siamese.batch_infer(TestTextSimilarity.src_text_ids)[0]

        self.distance_rank(
                text_emb_list,
                TestTextSimilarity.src_text_list,
                TestTextSimilarity.src_label_list,
                infer_res_path,
                topk)

    def test_textcnn_siamese_train(self):
        textcnn_config = {
                "vocab_size": TestTextSimilarity.tokenizer.size(),
                "emb_dim" : 512,
                "num_filters": 256,
                "num_channels":1,
                "win_size_list": [3],
                "is_sparse": True,
                "use_cudnn": True,
                "triplet_margin": 1.1,
                }

        run_config = {
                "model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "textcnn"),
                "best_model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "textcnn_best"),
                "epochs": 15,
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                "learning_rate": 5e-5,
                "load_best_model": False,
                }

        with D.guard():
            textcnn_siamese = TextCNNSiameseModel()
            textcnn_siamese.build(**textcnn_config)
            best_acc = textcnn_siamese.train(
                    TestTextSimilarity.train_data, TestTextSimilarity.eval_data,
                    **run_config)
        logging.info("ernie siamese est train score: {}".format(best_acc))

    def test_textcnn_siamese_infer(self):
        textcnn_config = {
                "vocab_size": TestTextSimilarity.tokenizer.size(),
                "emb_dim" : 512,
                "num_filters": 256,
                "num_channels":1,
                "win_size_list": [3],
                "is_sparse": True,
                "use_cudnn": True,
                "triplet_margin": 1.1,
                }

        infer_config = {
                "best_model_save_path": os.path.join(TestTextSimilarity.test_output_dir, "textcnn_best"),
                "batch_size": 32,
                "max_seq_len": 300,
                "print_step": 200,
                }

        topk=10

        infer_res_path = os.path.join(TestTextSimilarity.test_output_dir, "textcnn_infer_res.txt")

        with D.guard():
            textcnn_siamese = TextCNNSiameseModel()
            textcnn_siamese.build(**textcnn_config)
            textcnn_siamese.load_model(infer_config["best_model_save_path"])
            text_emb_list = textcnn_siamese.batch_infer(TestTextSimilarity.src_text_ids)[0]

        self.distance_rank(
                text_emb_list,
                TestTextSimilarity.src_text_list,
                TestTextSimilarity.src_label_list,
                infer_res_path,
                topk)

    def distance_rank(self, emb_list, text_list, label_list, rank_res_path, topk=10):
        # 计算各text对之间的距离
        dist = DistanceMetric.get_metric('euclidean')
        dist_matrix = dist.pairwise(emb_list)

        with codecs.open(rank_res_path, "w", "gb18030") as wf:
            # 给出每个文本的top5相似文本
            for index, (text, label) in enumerate(zip(text_list, label_list)):
                cur_distances = dist_matrix[index]
                # 从小到大排列的
                # 先找出topk近的索引
                topk_index_list = np.argpartition(cur_distances, topk-1)[:topk]
                #logging.info("cur topk index: {}".format(topk_index_list))
                #logging.info("cur topk distance: {}".format(cur_distances[topk_index_list]))
                topk_rerank = np.argsort(cur_distances[topk_index_list])
                #logging.info("cur topk distance rank ind: {}".format(topk_rerank))
                # 然后对topk距离的精确排序
                wf.write("text #%d: %s, label: %s\n" % (index, text, label))
                for cur_rank, rerank_ind in enumerate(topk_rerank):
                    cur_index = topk_index_list[rerank_ind]
                    wf.write("top #%d similiar: label: %s, score: %f, text: %s\n" % (
                        cur_rank,
                        TestTextSimilarity.src_label_list[cur_index],
                        dist_matrix[index][cur_index],
                        TestTextSimilarity.src_text_list[cur_index],
                        ))


if __name__ == "__main__":
    # 运行所有测试用例
    #unittest.main()

    # 运行指定测试用例
    # 构造测试集
    suit = unittest.TestSuite()
    #suit.addTest(TestTextSimilarity("test_textcnn_siamese_train"))
    #suit.addTest(TestTextSimilarity("test_textcnn_siamese_infer"))
    suit.addTest(TestTextSimilarity("test_ernie_siamese_train"))
    suit.addTest(TestTextSimilarity("test_ernie_siamese_infer"))
    runner = unittest.TextTestRunner()
    runner.run(suit)

