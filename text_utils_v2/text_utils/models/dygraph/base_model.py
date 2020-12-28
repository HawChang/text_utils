#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   base_model.py
Author:   zhanghao55@baidu.com
Date  :   20/12/21 10:47:13
Desc  :   
"""

import os
import sys
import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
import time

from sklearn.metrics import classification_report


def model_parallelized(strategy=None):
    if strategy is None:
        strategy = D.prepare_context()
    def set_parallelized(func):
        def wrapper(self, *args, **kwargs):
            logging.info("set model parallelized")
            func(self, *args, **kwargs)
            self.model = D.DataParallel(self.model, strategy)
            self.parallelized = True
        return wrapper
    return set_parallelized


def gen_batch_data(data_iter, batch_size=32, max_seq_len=300, max_ensure=False):
    assert batch_size > 0, "batch_size should be greater than 0, actual= {}".format(batch_size)
    batch_data = list()

    def pad(data_list):
        """对data_list各元素pad到等长
           若元素不可pad 返回None
        """
        # 处理样本
        # 确定当前data_list是否可以pad
        # 当列表的元素非列表或元组等具有长度的对象时 无法pad
        # 返回None 表示pad失败
        cur_max_len = 0
        for cur_data in data_list:
            try:
                if len(cur_data) > cur_max_len:
                    cur_max_len = len(cur_data)
            except TypeError as e:
                return None

        # 如果指定长度 则替换为最大长度
        # 否则最长为指定的max_seq_len
        if max_ensure:
            cur_max_len = max_seq_len
        else:
            cur_max_len = max_seq_len if cur_max_len > max_seq_len else cur_max_len

        # padding
        return [np.pad(x[:cur_max_len], [0, cur_max_len-len(x[:cur_max_len])], mode='constant') for x in data_list]

    def batch_process(cur_batch_data, cur_batch_size):
        # cur_batch_data为列表 且不为空
        # 元素均为tuple
        #logging.info("cur_batch_data[0]: {}".format(cur_batch_data[0]))
        #logging.info("cur_batch_data[0] type: {}".format(type(cur_batch_data[0])))
        #logging.info("cur_batch_data[0] len: {}".format(len(cur_batch_data[0])))
        batch_list = list()
        data_lists = list(zip(*cur_batch_data))
        #logging.info("zip(*cur_batch_data): {}".format(zip(*cur_batch_data)))
        #logging.info("list(zip(*cur_batch_data)): {}".format(data_lists))

        for index, data_list in enumerate(data_lists):
            #logging.info("data_list type: {}".format(type(data_list)))
            #logging.info("data_list: {}".format(data_list))
            padded_data_list = pad(data_list)
            if padded_data_list is None:
                #logging.info("input #{} pad fail, skipped".format(index))
                padded_data_list = data_list

            data_np = np.array(padded_data_list).reshape([cur_batch_size, -1])
            batch_list.append(data_np)

        return batch_list

    for data in data_iter:
        # 规定:如果有多个输入要batch 则用zip组合各输入列表
        # 则输入的data_iter的元素为tuple时，应该视为多输入
        # 当输入不是的时候，视为单输入，将其处理为tuple
        if not isinstance(data, tuple):
            data = (data,)

        if len(batch_data) == batch_size:
            # 当前已组成一个batch
            yield batch_process(batch_data, batch_size)
            batch_data = list()
        batch_data.append(data)

    if len(batch_data) > 0:
        yield batch_process(batch_data, len(batch_data))

class BaseModel(object):
    def __init__(self):
        """初始化
        """
        self.parallelized = False
        self.built = False

    def init_optimizer(self, learning_rate):
        """初始化优化器
        """
        if not self.built:
            raise RuntimeError("model should be built before get optimizer")

        self.optimizer = F.optimizer.Adam(
                learning_rate=learning_rate,
                parameter_list=self.model.parameters())

    def save_model(self, save_path):
        """保存模型
        """
        start_time = time.time()
        if self.parallelized and D.parallel.Env().local_rank == 0:
            F.save_dygraph(self.model.state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path):
        """加载模型
        """
        if os.path.exists(model_path + ".pdparams"):
            logging.info("load model from {}".format(model_path))
            start_time = time.time()
            sd, _ = D.load_dygraph(model_path)
            self.model.set_dict(sd)
            logging.info("cost time: %.4fs" % (time.time() - start_time))
        else:
            logging.info("cannot find model file: {}".format(model_path + ".pdparams"))

    def train(self, train_data_list, eval_data_list,
            model_save_path=None, best_model_save_path=None,
            epochs=5, batch_size=32, learning_rate=5e-5, max_seq_len=300,
            max_ensure=False, print_step=50, load_best_model=True,
            **kwargs):
        """ 训练dygraph模型
        [IN]  model: dygraph模型结构
              optimizer: 优化器
              train_data_list: list[(input1[, input2, ...], label)], 训练数据
              eval_data_list: list[(input1[, input2, ...], label)], 评估数据
              label_encoder: LabelEncoder, 类别转化工具
              model_save_path: string, 模型存储路径
              best_model_save_path: string, 最优模型存储路径
              epochs:  int, 训练轮数
              batch_size: int, 批大小
              max_seq_len: int, 最大长度
              max_ensure: boolean, true则始终补齐到max_seq_len
              best_acc: float, 最优acc初始值
              print_step: int, 每个print_step打印训练情况
              logits_softmax: boolean, true则验证时输出softmax后的logits
              eval_method: str, eval模型效果
              with_label: boolean, true则数据中有label
        [OUT] best_acc: float, 训练得到的最优acc
        """
        logging.info("train model start")
        train_start_time = time.time()
        # 加载最优模型
        if load_best_model:
            self.load_model(best_model_save_path)
        # 进入train模式
        self.model.train()
        # 初始化优化器
        self.init_optimizer(learning_rate)

        def train_data_reader():
            return  gen_batch_data(train_data_list, batch_size, max_seq_len, max_ensure)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            # 每个epoch都shuffle数据以获得最佳训练效果；
            np.random.shuffle(train_data_list)
            train_data_batch = F.contrib.reader.distributed_batch_reader(train_data_reader)() \
                    if self.parallelized else train_data_reader()
            for cur_train_batch in train_data_batch:
                cur_train_step += 1
                cur_train_batch = [D.to_variable(x) for x in cur_train_batch]
                loss = self.get_loss(*cur_train_batch, **kwargs)
                if self.parallelized:
                    # 若多卡 则将各训练的loss归一化
                    loss = self.model.scale_loss(loss)
                # 反向传播
                loss.backward()
                if self.parallelized:
                    # 若多卡 则各训练的梯度收集
                    # 注意梯度更新时需要时LoDTensor，即为dense矩阵
                    # 例如:embedding层的is_sparse参数需要为False,
                    #      否则更新时将是稀疏更新, 多卡训练时会出错
                    self.model.apply_collective_grads()
                self.optimizer.minimize(loss)
                # 清空梯度
                self.model.clear_gradients()
                if cur_train_step % print_step == 0:
                    speed = cur_train_step / (time.time() - train_start_time)
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss.numpy(), speed))

            if model_save_path is not None:
                # 每轮保存模型
                logging.info("save model at epoch {}".format(cur_epoch))
                self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # 计算验证集准确率
            cur_eval_res = self.evaluate(eval_data_list, batch_size=batch_size, max_seq_len=max_seq_len, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if best_model_save_path is not None and is_best:
                # 如果是当前最优效果模型 则保存为best模型
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def infer(self, infer_data_list, **kwargs):
        """ 用dygraph模型预测
        [IN]  model: dygraph模型结构
              infer_data_list: list[(input1[, input2, ...])], 待预测数据
        [OUT] pred: list[float], 预测结果
        """
        # 输入数据是否已转为paddle接收的tensor
        is_tensor = kwargs.pop("is_tensor", True)

        # 在这个with域内ernie不会进行梯度计算；
        with D.base._switch_tracer_mode_guard_(is_train=False):
            # 控制模型进入eval模式，这将会关闭所有的dropout；
            self.model.eval()
            # 如果infer_data_list没有转tensor 则转为paddle接收的tensor
            if not is_tensor:
                infer_data_list = [D.to_variable(np.array(x)) for x in infer_data_list]

            infer_res = self.model(*infer_data_list, **kwargs)

            # 按各输出聚合结果
            if isinstance(infer_res, tuple):
                infer_res = tuple([x.numpy() for x in infer_res])
            else:
                infer_res = infer_res.numpy()

            # 进入train模式
            self.model.train()
        return infer_res

    def batch_infer(self, infer_data_iter, batch_size=32, max_seq_len=300, max_ensure=False,
            print_step=20, **kwargs):
        """ 用dygraph模型逐批预测
        [IN]  model: dygraph模型结构
              infer_data_iter: iterable[(input1[, input2, ...])], 待预测数据
              with_label: boolean, true则infer_data_iter中为(数据,标签)二元组列表
              batch_size: int, 批大小
              max_seq_len: int, 最大长度
              print_step: int, 每个print_step打印训练情况
              logits_softmax: boolean, true则预测结果为softmax后的logits
        [OUT] pred: tuple(list[float]), 预测结果
        """
        infer_res_list = None

        # infer data不打乱
        infer_data_batch = gen_batch_data(
                infer_data_iter,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                max_ensure=max_ensure,
                )

        cur_infer_step = 0
        cur_infer_time = time.time()
        for cur_infer_batch in infer_data_batch:
            #for index, x in enumerate(cur_infer_batch):
                #logging.info("input #{}: {}".format(index, x))
                #logging.info("input #{} dtype: {}".format(index, x.dtype))
            cur_infer_batch = [D.to_variable(x) for x in cur_infer_batch]
            cur_infer_step += 1
            #logging.info("cur batch size = {}".format(len(cur_infer_batch[0])))
            cur_logits_list = self.infer(cur_infer_batch, **kwargs)
            #logging.info("cur logits: {}".format(cur_logits))
            #logging.info("cur logits type : {}".format(type(cur_logits_list)))
            #logging.info("cur logits len : {}".format(len(cur_logits_list)))

            if not isinstance(cur_logits_list, tuple):
                cur_logits_list = (cur_logits_list,)

            if infer_res_list is None:
                infer_res_list = list()
                for _ in range(len(cur_logits_list)):
                    infer_res_list.append(list())

            for output_ind, cur_logits in enumerate(cur_logits_list):
                infer_res_list[output_ind].extend(cur_logits)

            if cur_infer_step % print_step == 0:
                cost_time = time.time() - cur_infer_time
                speed = cur_infer_step / cost_time
                logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_infer_step, cost_time, speed))

        return tuple(infer_res_list)


    def build(self, *args, **kwargs):
        """网络构建函数
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        """模型训练阶段调用
        """
        raise NotImplementedError

    def evaluate(self, eval_data_list, batch_size=32, max_seq_len=300, **kwargs):
        """模型评估
        """
        raise NotImplementedError

    def check_if_best(self, cur_eval_res):
        """根据评估结果 判断是否最优
        """
        raise NotImplementedError

    def get_best_score(self):
        """
        """
        raise NotImplementedError


class ClassificationModel(BaseModel):
    def __init__(self):
        """初始化
        """
        super(ClassificationModel, self).__init__()
        self.best_acc = None

    def get_loss(self, *input_list, **kwargs):
        input_label = input_list[-1]
        input_data = input_list[:-1]
        loss = self.model(*input_data, labels=input_label, **kwargs)
        # 模型的返回值可能由多个 规定第一个为loss
        if isinstance(loss, tuple):
            loss = loss[0]
        return loss

    def train(self, *args, **kwargs):
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(ClassificationModel, self).train(*args, **kwargs)

    def evaluate(self, eval_list, batch_size=32, max_seq_len=300, print_step=50, **kwargs):
        eval_list = zip(*eval_list)
        eval_data = zip(*eval_list[:-1])
        all_logits = self.batch_infer(eval_data,
                labels=None,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                print_step=print_step,
                **kwargs)

        all_pred = np.argmax(all_logits[0], axis=-1)
        all_label = eval_list[-1]
        if self.label_encoder is not None:
            all_pred = [self.label_encoder.inverse_transform(x) for x in all_pred]
            all_label = [self.label_encoder.inverse_transform(x) for x in  all_label]

        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
        acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
        logging.info("eval acc : {}".format(acc))
        return acc

    def check_if_best(self, cur_eval_res):
        """根据评估结果判断是否最优
        """
        if self.best_acc is None or self.best_acc <= cur_eval_res:
            self.best_acc = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        return self.best_acc


class SiameseModel(BaseModel):
    def __init__(self):
        """初始化
        """
        super(SiameseModel, self).__init__()
        self.min_loss = None

    def get_loss(self, *input_list, **kwargs):
        loss = self.model(*input_list, **kwargs)
        # 模型的返回值可能由多个 规定第一个为loss
        if isinstance(loss, tuple):
            loss = loss[0]
        return loss

    def evaluate(self, eval_list, batch_size=32, max_seq_len=300, print_step=50, **kwargs):
        all_logits = self.batch_infer(eval_list,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                print_step=print_step,
                **kwargs)

        # loss是输出的第一个
        # 是个列表
        loss = np.mean(all_logits[0])
        logging.info("eval loss : {}".format(loss))
        return loss

    def check_if_best(self, cur_eval_res):
        """根据评估结果判断是否最优
        """
        if self.min_loss is None or self.min_loss >= cur_eval_res:
            self.min_loss = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        return self.min_loss
