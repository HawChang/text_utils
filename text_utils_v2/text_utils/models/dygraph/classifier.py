#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   classifier.py
Author:   zhanghao55@baidu.com
Date  :   20/12/14 18:47:05
Desc  :   
"""

import os
import sys
import codecs
import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as D

from text_utils.models.dygraph.nets.ernie_for_sequence_classification import ErnieModelCustomized
from text_utils.models.dygraph.nets.gru import GRU
from text_utils.models.dygraph.nets.textcnn import TextCNN
from text_utils.models.dygraph.train_infer_utils import batch_infer
from text_utils.models.dygraph.train_infer_utils import eval as dygraph_eval
from text_utils.models.dygraph.train_infer_utils import infer as dygraph_infer
from text_utils.models.dygraph.train_infer_utils import train
from text_utils.models.dygraph.utils.data_io import get_data, load_model

#_cur_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(_cur_dir, "../"))
from text_utils.tokenizers.ernie_tokenizer import ErnieTokenizer
from text_utils.utils.label_encoder import LabelEncoder


def textcnn_init(model_conf):
    """根据配置参数初始化模型
    """
    textcnn_params = {
            "num_class": model_conf.getint("num_class"),
            "vocab_size": model_conf.getint("vocab_size"),
            }

    if "emb_dim" in model_conf:
        textcnn_params["emb_dim"] = model_conf.getint("emb_dim")

    if "num_filters" in model_conf:
        textcnn_params["num_filters"] = model_conf.getint("num_filters")

    if "fc_hid_dim" in model_conf:
        textcnn_params["fc_hid_dim"] = model_conf.getint("fc_hid_dim")

    if "num_channels" in model_conf:
        textcnn_params["num_channels"] = model_conf.getint("num_channels")

    if "win_size_list" in model_conf:
        textcnn_params["win_size_list"] = [int(x) for x in model_conf["win_size_list"].split("|")]

    if "is_sparse" in model_conf:
        textcnn_params["is_sparse"] = model_conf.getboolean("is_sparse")

    if "use_cudnn" in model_conf:
        textcnn_params["use_cudnn"] = model_conf.getboolean("use_cudnn")

    return TextCNN(**textcnn_params)


def gru_init(model_conf):
    """根据配置参数初始化模型
    """
    gru_params = {
            "num_class": model_conf.getint("num_class"),
            "vocab_size": model_conf.getint("vocab_size"),
            }

    if "emb_dim" in model_conf:
        gru_params["emb_dim"] = model_conf.getint("emb_dim")

    if "gru_dim" in model_conf:
        gru_params["gru_dim"] = model_conf.getint("gru_dim")

    if "fc_hid_dim" in model_conf:
        gru_params["fc_hid_dim"] = model_conf.getint("fc_hid_dim")

    if "is_sparse" in model_conf:
        gru_params["is_sparse"] = model_conf.getboolean("is_sparse")

    if "bi_direction" in model_conf:
        gru_params["bi_direction"] = model_conf.getboolean("bi_direction")

    return GRU(**gru_params)


def ernie_init(model_conf):
    """根据配置参数初始化模型
    """
    ernie_params = {
            "pretrain_dir_or_url": model_conf["pretrain_path"],
            "num_labels": model_conf.getint("num_class"),
            }
    return ErnieModelCustomized.from_pretrained(**ernie_params)


class Classifier(object):
    """paddle动态图分类器
    """
    def __init__(self):
        """初始化
        """
        self.model_dict = {
                "textcnn": textcnn_init,
                "gru": gru_init,
                "ernie": ernie_init,
                }

    def init_train(self, train_file_path, test_file_path, output_dir, \
            model_type, model_dir, model_conf):
        """训练阶段初始化
        """
        def gen_data_iter(data_path):
            """生成该文件数据的迭代器
            """
            def line_processor(line):
                """每行文本的处理函数
                """
                parts = line.strip("\n").split("\t")
                label = parts[0]
                text = parts[1]
                return (text, label)

            return get_data(data_path, read_func=line_processor)

        #train_data = list(get_data(self.train_file_path, read_func=line_processor))
        #eval_data = get_data(self.test_file_path, read_func=line_processor)

        _, train_label_list = zip(*gen_data_iter(train_file_path))

        self.label_encoder = LabelEncoder({x:int(x) for x in train_label_list}, isFile=False)
        self.label_encoder.save(os.path.join(model_dir, model_conf["label_id_path"]))

        with D.guard():
            self.model, self.tokenizer = self.init_model(
                    model_type, model_dir, model_conf, self.label_encoder.size())

            self.optimizer = F.optimizer.Adam(
                        learning_rate=model_conf.getfloat("learning_rate"),
                        parameter_list=self.model.parameters())
            logging.info("learning_rate : {}".format(model_conf.getfloat("learning_rate")))

        self.train_data = [(self.tokenizer.encode(x[0])[0], int(x[1])) for x in gen_data_iter(train_file_path)]
        self.eval_data = [(self.tokenizer.encode(x[0])[0], int(x[1]) )for x in gen_data_iter(test_file_path)]

        logging.info("train_data size: {}".format(len(self.train_data)))
        logging.info("test_data size: {}".format(len(self.eval_data)))

    def init_infer(self, model_type, model_dir, model_conf):
        """预测阶段初始化
        """
        self.label_encoder = LabelEncoder(
                os.path.join(model_dir, model_conf["label_id_path"]))

        with D.guard():
            self.model, self.tokenizer = self.init_model(
                    model_type, model_dir, model_conf, self.label_encoder.size())

    def init_model(self, model_type, model_dir, model_conf, num_class):
        """模型初始化:模型和tokenizer
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.model_conf = model_conf

        tokenizer = ErnieTokenizer.from_pretrained(model_conf["vocab_path"])

        model_conf["num_class"] = str(num_class)
        model_conf["vocab_size"] = str(tokenizer.size())

        model = self.model_dict[model_type](model_conf)

        load_model(model, os.path.join(model_dir, model_conf["best_model_path"]))
        return model, tokenizer

    def train(self):
        """训练
        """
        with D.guard():
            best_acc = train(self.model, self.optimizer,
                    self.train_data, self.eval_data, self.label_encoder, best_acc=0,
                    model_save_path=os.path.join(self.model_dir, self.model_conf["model_save_path"]),
                    best_model_save_path=os.path.join(self.model_dir, self.model_conf["best_model_path"]),
                    epochs=self.model_conf.getint("epoch"),
                    batch_size=self.model_conf.getint("batch_size"),
                    max_seq_len=self.model_conf.getint("max_seq_len"),
                    print_step=self.model_conf.getint("print_step"),
                    )
        logging.info("{} best train score: {}".format(self.model_type, best_acc))

    def test(self):
        """测试
        """
        with D.guard():
            acc = dygraph_eval(self.model, self.eval_data, self.label_encoder)
        logging.info("{} final test score: {}".format(self.model_type, acc))

    def check(self, text):
        """预测结果
        text: unicode
        """
        text_id = self.tokenizer.encode(text)[0]
        with D.guard():
            logits = dygraph_infer(self.model, [text_id], is_tensor=False)[0]
        pred_label_id = np.argmax(logits)

        pred_label_name = self.label_encoder.inverse_transform(pred_label_id)
        pred_rate = logits[pred_label_id]
        return pred_label_name, pred_rate

    def infer(self, infer_file, output_file, batch_size=32):
        """推理，和check的区别是infer是针对文件的预测，而check是针对单条文本的预测
        Args:
            infer_file:     待预测文件，必须是两列以上，其中第二列为文本
            output_file:    预测文件，第一列为模型预测文件
        """
        def line_processor(line):
            """每行文本的处理函数
            """
            parts = line.strip("\n").split("\t")
            text = parts[1]
            return (self.tokenizer.encode(text)[0], text)

        # batch_infer预测时，data_iter可带上其标签信息，这里把标签信息替换成物料文本
        # 方便之后结果输出
        infer_data_iter = get_data(infer_file, read_func=line_processor)
        with D.guard():
            pred_logits, text_list = batch_infer(self.model, infer_data_iter, batch_size=batch_size)

        pred_label_id = np.argmax(pred_logits, axis=-1)
        pred_label_name = [self.label_encoder.inverse_transform(x) for x in pred_label_id]

        with codecs.open(output_file, "w", 'gb18030') as wf:
            for label, text in zip(pred_label_name, text_list):
                wf.write("%s\t%s\n" % (label, text))


if __name__ == "__main__":
    pass
