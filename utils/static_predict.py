#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   static_predict.py
Author:   zhanghao55@baidu.com
Date  :   20/08/25 14:32:44
Desc  :   
"""

import logging
import numpy as np
import sys
import time

from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from ernie_tokenizer import ErnieTokenizer
from label_encoder import LabelEncoder

import configparser
import datetime

class StaticPredictor(object):
    """静态图预测类
    """
    def __init__(self, model_dir, label_id_path, vocab_path,
            gpu_id=None, gpu_mem=8000, zero_copy=True):
        self.tokenizer = ErnieTokenizer.from_pretrained(vocab_path)
        self.id_2_token = {v: k for k, v in self.tokenizer.vocab.items()}

        label_encoder = LabelEncoder(label_id_info=label_id_path, isFile=True)
        self.id_label_dict = label_encoder.id_label_dict

        # 设置AnalysisConfig
        config = AnalysisConfig(model_dir)
        if gpu_id is None:
            config.disable_gpu()
        else:
            config.enable_use_gpu(gpu_mem, gpu_id)
            logging.info("gpu id: {}".format(config.gpu_device_id()))

        self.zero_copy = zero_copy
        if self.zero_copy:
            config.switch_use_feed_fetch_ops(False)

        # 创建PaddlePredictor
        self.predictor = create_paddle_predictor(config)

        if self.zero_copy:
            input_names = self.predictor.get_input_names()
            #logging.info(input_names)
            self.input_tensor = self.predictor.get_input_tensor(input_names[0])

            output_names = self.predictor.get_output_names()
            #logging.info(output_names)
            self.output_tensor = self.predictor.get_output_tensor(output_names[0])

    def predict_proba(self, text_list, batch_size=32, max_seq_len=300):
        """预测 返回概率
        """
        predict_time = 0
        tokenize_time = 0
        res_list = list()
        for cur_batch_data_ids, cur_tokenize_time in \
                self.batch(text_list, batch_size, max_seq_len, max_ensure=False):
            tokenize_time += cur_tokenize_time
            start_time = time.time()
            if self.zero_copy:
                self.input_tensor.copy_from_cpu(np.array(cur_batch_data_ids))
                self.predictor.zero_copy_run()
                logits = self.output_tensor.copy_to_cpu()
            else:
                data_tensor = [PaddleTensor(np.array(cur_batch_data_ids))]
                logits = self.predictor.run(data_tensor)[0].as_ndarray()
            predict_time += time.time() - start_time

            res_list.append(logits)
        logging.info("predict time: %.4fs, tokenize_time: %.4fs"\
                % (predict_time, tokenize_time))
        return np.concatenate(res_list, axis=0)

    def predict(self, text_list, batch_size=32, max_seq_len=300):
        """预测
        """
        res_list = self.predict_proba(text_list, batch_size, max_seq_len)
        pred_label_id_list = np.argmax(res_list, axis=-1).tolist()
        rate_list = res_list[range(len(pred_label_id_list)), pred_label_id_list]
        label_list = [self.id_label_dict[x] for x in pred_label_id_list]

        return zip(label_list, rate_list)

    def _gen_batch_data(self, data_iter, batch_size=32):
        """数据分批
        """
        batch_data = list()
        for data in data_iter:
            if len(batch_data) == batch_size:
                # 当前已组成一个batch
                yield batch_data
                batch_data = list()
            batch_data.append(data)

        if len(batch_data) > 0:
            yield batch_data

    def batch(self, data_iter, batch_size=32,
            max_seq_len=300, max_ensure=False):
        """生成成批处理后的数据
        """
        batch_data = list()

        def pad(data_list):
            """补齐
            """
            # 处理样本
            # 确定当前批次最大长度
            if max_ensure:
                cur_max_len = max_seq_len
            else:
                cur_max_len = max([len(x) for x in data_list])
                cur_max_len = max_seq_len if cur_max_len > max_seq_len else cur_max_len

            # padding
            return [np.pad(x[:cur_max_len], [0, cur_max_len - len(x[:cur_max_len])], \
                    mode='constant') for x in data_list]

        def batch_process(cur_batch_data, cur_batch_size):
            """批处理
            """
            start_time = time.time()
            cur_batch_data_ids = [self.tokenizer.encode(x)[0] for x in cur_batch_data]
            return pad(cur_batch_data_ids), time.time() - start_time

        for cur_batch in self._gen_batch_data(data_iter, batch_size):
            yield batch_process(cur_batch, len(cur_batch))


def stream_predict(instream, config_path, uniqid=None, encoding="utf-8"):
    """运行入口函数
    """
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)
    if uniqid is not None:
        if uniqid == "{time}":
            now_time = datetime.datetime.now()
            #uniqid = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")
            uniqid = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")
        # 添加当前uniqid
        config.set("DEFAULT", "uniqid", uniqid)
    logging.info("uniqid: {}".format(config.get("DEFAULT", "uniqid")))

    model_config = config["MODEL_PATH"]
    static_model_path = model_config["static_textcnn_model"]
    label_id_path = model_config["label_encoder"]
    vocab_path = model_config["tokenizer"]

    model = StaticPredictor(
            static_model_path,
            label_id_path,
            vocab_path,
            gpu_id=config.getint("RUN", "cuda_visible_devices"),
            zero_copy=True)

    for line in instream:
        text = line.strip("\n").decode(encoding)
        pred_label, pred_rate = model.predict([text])[0]
        print("\t".join([pred_label, str(pred_rate), text]).encode("utf-8"))


if __name__ == "__main__":
    config_path = sys.argv[1]
    uniqid = sys.argv[2] if len(sys.argv) > 2 else None
    stream_predict(sys.stdin, config_path, uniqid)
