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
import time
import torch

from tqdm import tqdm

def model_distributed(local_rank=None, find_unused_parameters=False):
    if local_rank is None:
        local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info("set device {} to rank {}".format(device, local_rank))
    def set_distributed(func):
        def wrapper(self, *args, **kwargs):
            logging.info("set model distributed")
            self.device = device
            self.local_rank = local_rank
            self.is_master = False if local_rank != 0 else True
            model = func(self, *args, **kwargs)
            model.to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=find_unused_parameters)
            self.distributed = True
            return model
        return wrapper
    return set_distributed


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        """初始化
        """
        # 分布式训练时为True
        self.distributed = False
        # 当分布式训练时 local_rank为各进程唯一ID 为0的为主进程
        # 当单机单卡训练时 local_rank为0
        self.local_rank = 0
        # 当分布式训练 但该进程不是主进程时 is_master为False，其余情况均为True
        self.is_master = True
        self.model = self.init_model(*args, **kwargs)
        if not self.distributed:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def init_optimizer(self, learning_rate, weight_decay=1e-3, *args, **kwargs):
        """初始化优化器
        """
        return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

    def save_model(self, save_path):
        """保存模型
        """
        start_time = time.time()
        torch.save(self.get_model().state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path, strict=True):
        """加载模型
        """
        if os.path.exists(model_path):
            logging.info("load model from {}".format(model_path))
            start_time = time.time()
            state_dict = torch.load(model_path)
            logging.debug("state_dict_names: {}".format(state_dict.keys()))
            self.get_model().load_state_dict(state_dict, strict=strict)
            torch.cuda.empty_cache()
            logging.info("cost time: %.4fs" % (time.time() - start_time))
        else:
            logging.info("cannot find model file: {}".format(model_path))

    def get_model(self):
        """取得模型
        """
        if self.distributed:
            return self.model.module
        else:
            return self.model

    def train(self, train_dataloader, eval_dataloader,
            model_save_path=None, best_model_save_path=None,
            epochs=5, learning_rate=5e-5,
            print_step=50, load_best_model=True,
            strict=True, **kwargs):
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
        logging.info("train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()
        # 加载最优模型
        if load_best_model:
            self.load_model(best_model_save_path, strict)
        # 进入train模式
        self.model.train()
        # 初始化优化器
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            for cur_train_batch in tqdm(train_dataloader):
                cur_train_step += 1
                loss = self.get_loss(cur_train_batch)

                #cur_train_batch = [x.to(self.device) for x in cur_train_batch]
                #loss = self.get_loss(*cur_train_batch, **kwargs)

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

                if cur_train_step % print_step == 0:
                    speed = cur_train_step / (time.time() - train_start_time)
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss.cpu().detach().numpy(), speed))

            if self.is_master and model_save_path is not None:
                # 每轮保存模型
                logging.info("save model at epoch {}".format(cur_epoch))
                self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # 计算验证集准确率
            cur_eval_res = self.evaluate(eval_dataloader, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and best_model_save_path is not None and is_best:
                # 如果是当前最优效果模型 则保存为best模型
                logging.warning("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def init_model(self, *args, **kwargs):
        """网络构建函数
        """
        raise NotImplementedError

    def get_loss(self, batch):
        """训练时如何得到loss
        """
        raise NotImplementedError

    def evaluate(self, eval_dataloader, **kwargs):
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


#class ClassificationModel(BaseModel):
    #def infer(self, infer_data_list, **kwargs):
    #    """ 用dygraph模型预测
    #    [IN]  model: dygraph模型结构
    #          infer_data_list: list[(input1[, input2, ...])], 待预测数据
    #    [OUT] pred: list[float], 预测结果
    #    """
    #    # 输入数据是否已转为paddle接收的tensor
    #    is_tensor = kwargs.pop("is_tensor", True)

    #    # 在这个with域内ernie不会进行梯度计算；
    #    with D.base._switch_tracer_mode_guard_(is_train=False):
    #        # 控制模型进入eval模式，这将会关闭所有的dropout；
    #        self.model.eval()
    #        # 如果infer_data_list没有转tensor 则转为paddle接收的tensor
    #        if not is_tensor:
    #            infer_data_list = [D.to_variable(np.array(x)) for x in infer_data_list]

    #        infer_res = self.model(*infer_data_list, **kwargs)

    #        # 按各输出聚合结果
    #        if isinstance(infer_res, tuple):
    #            infer_res = tuple([x.numpy() for x in infer_res])
    #        else:
    #            infer_res = infer_res.numpy()

    #        # 进入train模式
    #        self.model.train()
    #    return infer_res

    #def batch_infer(self, infer_data_iter, batch_size=32, max_seq_len=300, max_ensure=False,
    #        print_step=20, **kwargs):
    #    """ 用dygraph模型逐批预测
    #    [IN]  model: dygraph模型结构
    #          infer_data_iter: iterable[(input1[, input2, ...])], 待预测数据
    #          with_label: boolean, true则infer_data_iter中为(数据,标签)二元组列表
    #          batch_size: int, 批大小
    #          max_seq_len: int, 最大长度
    #          print_step: int, 每个print_step打印训练情况
    #          logits_softmax: boolean, true则预测结果为softmax后的logits
    #    [OUT] pred: tuple(list[float]), 预测结果
    #    """
    #    infer_res_list = None

    #    # infer data不打乱
    #    infer_data_batch = gen_batch_data(
    #            infer_data_iter,
    #            batch_size=batch_size,
    #            max_seq_len=max_seq_len,
    #            max_ensure=max_ensure,
    #            )

    #    cur_infer_step = 0
    #    cur_infer_time = time.time()
    #    for cur_infer_batch in infer_data_batch:
    #        #for index, x in enumerate(cur_infer_batch):
    #            #logging.info("input #{}: {}".format(index, x))
    #            #logging.info("input #{} dtype: {}".format(index, x.dtype))
    #        cur_infer_batch = [D.to_variable(x) for x in cur_infer_batch]
    #        cur_infer_step += 1
    #        #logging.info("cur batch size = {}".format(len(cur_infer_batch[0])))
    #        cur_logits_list = self.infer(cur_infer_batch, **kwargs)
    #        #logging.info("cur logits: {}".format(cur_logits))
    #        #logging.info("cur logits type : {}".format(type(cur_logits_list)))
    #        #logging.info("cur logits len : {}".format(len(cur_logits_list)))

    #        if not isinstance(cur_logits_list, tuple):
    #            cur_logits_list = (cur_logits_list,)

    #        if infer_res_list is None:
    #            infer_res_list = list()
    #            for _ in range(len(cur_logits_list)):
    #                infer_res_list.append(list())

    #        for output_ind, cur_logits in enumerate(cur_logits_list):
    #            infer_res_list[output_ind].extend(cur_logits)

    #        if cur_infer_step % print_step == 0:
    #            cost_time = time.time() - cur_infer_time
    #            speed = cur_infer_step / cost_time
    #            logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
    #                    % (cur_infer_step, cost_time, speed))

    #    return tuple(infer_res_list)

class Seq2seqModel(BaseModel):
    def generate(self, *args, **kwargs):
        raise NotImplementedError

class BertSeq2seqModel(Seq2seqModel):
    def __init__(self, *args, **kwargs):
        """初始化
        """
        super(BertSeq2seqModel, self).__init__(*args, **kwargs)
        self.min_loss = None

    def get_loss(self, batch):
        token_ids, token_type_ids, target_ids = [x.to(self.device) for x in batch]
        _, loss = self.model(token_ids,
                token_type_ids,
                labels=target_ids,
                device=self.device)
        return loss

    def train(self, *args, **kwargs):
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(BertSeq2seqModel, self).train(*args, **kwargs)

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        for batch in eval_dataloader:
            cur_eval_step += 1
            loss = self.get_loss(batch)
            loss_list.append(loss.item())
            if cur_eval_step % print_step == 0:
                cost_time = time.time() - start_time
                speed = cur_eval_step / cost_time
                logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_eval_step, cost_time, speed))
        loss_mean = np.mean(loss_list)
        if self.distributed:
            # 当分布式训练时 如果要考虑全部的loss
            # 则如下操作
            if gather_loss:
                loss_tensor = torch.tensor(loss_mean).to(self.device)
                # 这里只打印master进程的loss 所以只需要reduce到rank为0的进程
                # 如果要所有进程loss_tensor同步 用all_reduce
                torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
                if self.is_master:
                    logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
                    loss_mean = loss_tensor / torch.distributed.get_world_size()
                    logging.infer("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
            elif self.is_master:
                # 否则只有master进程打印loss
                logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        else:
            logging.info("eval loss = {}.".format(loss_mean))

        if self.is_master:
            self.gen_poem(is_poem=False)

        return loss_mean

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

    def generate(self, text, beam_size, is_poem=True):
        return self.get_model().generate(text, beam_size=beam_size, device=self.device, is_poem=is_poem)

    def gen_poem(self, beam_size=3, is_poem=True):
        test_data = ["北国风光##五言绝句", "题西林壁##七言绝句", "长安早春##五言律诗"]
        for text in test_data:
            logging.info(text)
            logging.info(self.generate(text, beam_size=beam_size, is_poem=is_poem))
