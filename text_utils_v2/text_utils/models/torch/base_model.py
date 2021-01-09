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
from text_utils.models.torch.config import yayun_list

from sklearn.metrics import classification_report


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
            # 在cpu上加载数据 然后加载到模型
            # 不然在分布式训练时 各卡都会在cuda:0上加载一次数据
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
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

    def trial_train(self, train_dataloader, *args,
            model_save_path=None, best_model_save_path=None,
            epochs=5, learning_rate=5e-5,
            print_step=50, load_best_model=True,
            strict=True, **kwargs):
        """ 训练dygraph模型
        [IN]
              train_dataloader: DataLoader, 训练数据
              epochs:  int, 训练轮数
              learning_rate: float, 学习率
              print_step: int, 每个print_step打印训练情况
              load_best_model: bool
        [OUT] best_acc: float, 训练得到的最优acc
        """
        logging.info("trail train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()
        # 加载最优模型
        if load_best_model:
            self.load_model(best_model_save_path, strict)
        # 初始化优化器
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        trial_batch = next(iter(train_dataloader))
        cur_train_step = 0
        for cur_epoch in range(epochs):
            # 进入train模式
            # 每epoch都要train 因为evaluate的时候会变eval
            self.model.train()
            for _ in tqdm(range(print_step)):
                cur_train_step += 1
                loss = self.get_loss(*trial_batch)

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
            cur_eval_res = self.evaluate([trial_batch], **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and best_model_save_path is not None and is_best:
                # 如果是当前最优效果模型 则保存为best模型
                logging.warning("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

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
        # 初始化优化器
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            # 进入train模式
            # 每epoch都要train 因为evaluate的时候会变eval
            self.model.train()
            for cur_train_batch in tqdm(train_dataloader):
                cur_train_step += 1
                loss = self.get_loss(*cur_train_batch)

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

    def get_loss(self, *args, **kwargs):
        """训练时如何得到loss
        """
        raise NotImplementedError

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
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
    def __init__(self, best_acc=None, label_encoder=None, *args, **kwargs):
        """初始化
        """
        super(ClassificationModel, self).__init__(*args, **kwargs)
        self.best_acc = best_acc
        self.label_encoder = label_encoder

    def get_loss(self, *input_list, **kwargs):
        input_list = [x.to(self.device) for x in input_list]
        input_label = input_list[-1]
        input_data = input_list[:-1]
        logging.debug("input_list size: {}".format(len(input_list)))
        loss = self.model(*input_data, labels=input_label, **kwargs)
        # 模型的返回值可能由多个 规定第一个为loss
        if isinstance(loss, tuple):
            loss = loss[0]
        return loss

    def train(self, *args, **kwargs):
        return super(ClassificationModel, self).train(*args, **kwargs)

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        all_pred = list()
        all_label = list()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        for cur_eval_batch in eval_dataloader:
            cur_eval_step += 1
            cur_token_ids, cur_token_type_ids, cur_label_ids = cur_eval_batch
            cur_logits = self.infer(cur_token_ids, cur_token_type_ids, labels=None)
            cur_pred = np.argmax(cur_logits, axis=-1)
            cur_label = cur_label_ids.detach().numpy()
            all_pred.extend(cur_pred)
            all_label.extend(cur_label)

            if cur_eval_step % print_step == 0:
                cost_time = time.time() - start_time
                speed = cur_eval_step / cost_time
                logging.info('eval step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_eval_step, cost_time, speed))

        if self.label_encoder is not None:
            all_pred = [self.label_encoder.inverse_transform(x) for x in all_pred]
            all_label = [self.label_encoder.inverse_transform(x) for x in  all_label]

        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
        acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
        logging.warning("rank {} eval acc : {}".format(self.local_rank, acc))
        return acc

        #if self.distributed:
        #    # 当分布式训练时 如果要考虑全部的loss
        #    # 则如下操作
        #    #if gather_loss:
        #    #    loss_tensor = torch.tensor(loss_mean).to(self.device)
        #    #    # 这里只打印master进程的loss 所以只需要reduce到rank为0的进程
        #    #    # 如果要所有进程loss_tensor同步 用all_reduce
        #    #    torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
        #    #    if self.is_master:
        #    #        logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
        #    #        loss_mean = loss_tensor / torch.distributed.get_world_size()
        #    #        logging.infer("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
        #    #elif self.is_master:
        #    #    # 否则只有master进程打印loss
        #    #    logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        #    logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        #else:
        #    logging.info("eval loss = {}.".format(loss_mean))

        #return loss_mean

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

    def infer(self, *infer_data_list, **kwargs):
        """ 用dygraph模型预测
        [IN]  model: dygraph模型结构
              infer_data: list[(input1[, input2, ...])], 待预测数据
        [OUT] pred: list[float], 预测结果
        """
        # 输入数据是否已转为paddle接收的tensor
        is_tensor = kwargs.pop("is_tensor", True)

        # infer时不保存反向的梯度
        with torch.no_grad():
            # 控制模型进入eval模式，这将会关闭所有的dropout和norm；
            self.model.eval()
            # 如果infer_data_list没有转tensor 则转为torch接收的tensor
            if not is_tensor:
                infer_data_list = [torch.tensor(x, device=self.device) for x in infer_data_list]
            else:
                infer_data_list = [x.to(self.device) for x in infer_data_list]

            infer_res = self.model(*infer_data_list, **kwargs)

            # 按各输出聚合结果
            if isinstance(infer_res, tuple):
                infer_res = tuple([x.detach().cpu().numpy() for x in infer_res])
            else:
                infer_res = infer_res.detach().cpu().numpy()

        return infer_res

    def batch_infer(self, infer_dataloader, print_step=20, **kwargs):
        """ 用dygraph模型逐批预测
        [IN]  model: dygraph模型结构
              infer_dataloader: DataLoader, 待预测数据
              print_step: int, 每个print_step打印训练情况
              logits_softmax: boolean, true则预测结果为softmax后的logits
        [OUT] pred: tuple(list[float]), 预测结果
        """
        infer_res_list = None

        cur_infer_step = 0
        cur_infer_time = time.time()
        with torch.no_grad():
            self.model.eval()
            for cur_infer_tuple in infer_dataloader:
                if not isinstance(cur_infer_tuple, tuple):
                    cur_infer_tuple = (cur_infer_tuple,)
                cur_infer_step += 1
                cur_logits_tuple = self.infer(*cur_infer_tuple, **kwargs)

                if not isinstance(cur_logits_tuple, tuple):
                    cur_logits_tuple = (cur_logits_tuple,)

                if infer_res_list is None:
                    infer_res_list = list()
                    for _ in range(len(cur_logits_tuple)):
                        infer_res_list.append(list())

                for output_ind, cur_logits in enumerate(cur_logits_tuple):
                    infer_res_list[output_ind].extend(cur_logits.detach().numpy())

                if cur_infer_step % print_step == 0:
                    cost_time = time.time() - cur_infer_time
                    speed = cur_infer_step / cost_time
                    logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                            % (cur_infer_step, cost_time, speed))

        return tuple(infer_res_list)


class Seq2seqModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs["tokenizer"]
        super(Seq2seqModel, self).__init__(*args, **kwargs)

    def generate(self, text, out_max_length=40, beam_size=1, device="cpu", is_poem=False, max_length=256):
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length

        # print(text)
        # token_type_id 全为0
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=device).view(1, -1)
        if is_poem:## 古诗的beam-search稍有不同
            logging.debug("poem beam_search")
            out_puts_ids = self.beam_search_poem(
                    text,
                    token_ids,
                    token_type_ids,
                    self.tokenizer.token2id,
                    beam_size=beam_size,
                    device=device)
        else:
            logging.debug("common beam_search")
            out_puts_ids = self.beam_search(
                    token_ids,
                    token_type_ids,
                    self.tokenizer._token_sep_id,
                    beam_size=beam_size,
                    device=device)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def beam_search_poem(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        title = text.split("##")[0]
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = [[] for i in range(beam_size)]
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = None #torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                #logging.debug("step #{}".format(step))
                #logging.debug("last_chars: {}".format(last_chars))
                if step == 0:
                    scores = self.model(token_ids, token_type_ids, device=device)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.model(new_input_ids, new_token_type_ids, device=device)
    
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
    
                if last_chars is not None:
                    for i, char in enumerate(last_chars):
    
                        for word in repeat_word[i]:
                            logit_score[i, word] -= 5
                        for word in title:
                            ix = word2ix.get(word, -1)
                            if ix != -1:
                                logit_score[i, ix] += 2
    
                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    if last_chars is not None:
                        for i, char in enumerate(last_chars):
                            if yayun_chars[i].item() != -1:
                                yayuns = yayun_list[yayun_chars[i].item()]
                                for char in yayuns:
                                    ix = word2ix.get(char, -1)
                                    if ix != -1:
                                        # print("char is " + str(char))
                                        logit_score[i, ix] += 10
    
    
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
    
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    #logging.debug("cur_out: {}".format(ix2word[each_out]))
    
                    if each_out in repeat_word[index]:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word[index].append(each_out)
    
                    if start < beam_size and each_out == douhao_id and last_chars is not None:
                        start += 1
                        #logging.debug("last_chars[{}] = {}".format(index, last_chars[index]))
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break
    
                    # if each_out == juhao_id and len(last_chars) != 0:  
                    #     word = ix2word[last_chars[index].item()]
                    #     if yayun_chars[index].item() != -1 and word in yayun_list[yayun_chars[index].item()]:
                    #         hype_score[index] += 10
                    #     else:
                    #         hype_score[index] -= 5
    
                # 更新得分
                output_scores = hype_score
    
                last_chars = indice2
    
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
    
                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()
    
                        new_repeat_word = []
                        for index, i in enumerate(flag):
                            if i.item() == 1:
                                new_repeat_word.append(repeat_word[index])
    
                        repeat_word = new_repeat_word
    
    
            # print(repeat_word)
            # print(yayun_chars)
            return output_ids[output_scores.argmax()]

    def beam_search(self, token_ids, token_type_ids, stop_id, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        # 一次只输入一个
        # batch_size = 1

        # token_ids shape: [batch_size, seq_length]
        logging.debug("token_ids: {}".format(token_ids))
        logging.debug("token_ids shape: {}".format(token_ids.shape))

        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids : {}".format(token_type_ids))
        logging.debug("token_type_ids  shape: {}".format(token_type_ids.shape))

        #sep_id = word2ix["[SEP]"]

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        logging.debug("output_ids: {}".format(output_ids))
        logging.debug("output_ids shape: {}".format(output_ids.shape))
        # 用来保存累计得分

        with torch.no_grad():
            # 初始化各得分
            # output_scores shape: [batch_size]
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            # 重复生成 直到达到最大长度
            for step in range(self.out_max_length):
                logging.debug("beam size: {}".format(beam_size))
                if step == 0:
                    # score shape: [batch_size, seq_length, self.vocab_size]
                    scores = self.model(token_ids, token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                    # 重复beam-size次 输入ids
                    # token_ids shape: [beam_size, batch_size*seq_length]
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_ids shape: {}".format(token_ids.shape))

                    # token_type_ids shape: [beam_size, batch_size*seq_length]
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
                else:
                    # TODO score shape: [beam_size, cur_seq_length, self.vocab_size]
                    # cur_seq_length是逐渐变化的
                    scores = self.model(new_input_ids, new_token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                # 只取最后一个输出在vocab上的score
                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                # 这是beam_size种结果各vocab的结果打平
                logit_score = logit_score.view(-1)
                # 找到topk的值和位置
                hype_score, hype_pos = torch.topk(logit_score, beam_size)

                # 根据打平后的位置 找到其打平前的行列位置
                # 行位置其实是beam_size中 的第几个beam 行位置可能有重复
                # 列位置其实是当前beam下的vocab_id vocab_id也可能有重复
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                logging.debug("indice1: {}".format(indice1))
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                logging.debug("indice2: {}".format(indice2))

                # 更新得分
                # output_scores shape: [beam_size]
                output_scores = hype_score
                logging.debug("output_scores: {}".format(output_scores))

                # 更新output_ids
                # 通过indice1选是哪个beam
                # 通过indice2选当前beam加哪个vocab
                # output_ids shape: [beam_size, cur_seq_length]
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                logging.debug("output_ids: {}".format(output_ids))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # token_ids是固定原输入
                # output_ids是当前beam_search留下的beam_size个候选路径
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # output_ids的type全为1
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))

                # 记录当前output_ids中有sep_id的情况
                end_counts = (output_ids == stop_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 如果当前分数最优的已结束 则选当前该beam
                    # 说明出现终止了～
                    return output_ids[best_one]
                else :
                    # 否则 去除出现sep_id的序列
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        logging.debug("beam size change")

            return output_ids[output_scores.argmax()]


class BertSeq2seqModel(Seq2seqModel):
    def __init__(self, *args, **kwargs):
        """初始化
        """
        super(BertSeq2seqModel, self).__init__(*args, **kwargs)
        self.min_loss = None

    def get_loss(self, *batch):
        token_ids, token_type_ids, target_ids = [x.to(self.device) for x in batch]
        predictions, loss = self.model(token_ids,
                token_type_ids,
                labels=target_ids,
                device=self.device)

        ## predictions去除了对最后一个sep的预测结果
        ## predictions每个位置都是对token_ids的下个位置的预测
        ## 即对于某一个预测prediction[i]，其对应预测的是token_id[i+1]
        ## 因此我们需要的是prediction的范围，其实是token_type_ids左移一位，得到pred_mask
        ## 此时pred_mask中为1的是需要考虑的预测结果
        #pred_mask = token_type_ids[:, 1:].contiguous()

        ## 这里要生成诗的时候 对重复生成的id进行惩罚
        # TODO 实际效果并不好 每句诗都有重复两次的标点
        #logging.info("loss: {}".format(loss))
        #duplicate_loss = self.get_duplicate_loss(predictions, token_type_ids)
        #logging.info("duplicate_loss: {}".format(duplicate_loss))
        #loss += duplicate_loss

        return loss

    def get_duplicate_loss(self, predictions, token_type_ids):
        """
        """
        # predictions shape: [batch_size, seq_length, vocab_size]
        logging.debug("predictions shape: {}".format(predictions.shape))

        # max_values shape: [batch_size, seq_length]
        # max_inds shape: [batch_size, seq_length]
        max_values, max_inds = predictions.max(axis=-1)
        logging.debug("max_values shape: {}".format(max_values.shape))
        logging.debug("max_inds shape: {}".format(max_inds.shape))
        #logging.info("max_inds[:10]: {}".format(max_inds[:10]))

        # temp shape: [batch_size, seq_length, 1]
        temp = max_inds.unsqueeze(2)
        logging.debug("temp shape: {}".format(temp.shape))
        # temp_t shape: [batch_size, 1, seq_length]
        temp_t = max_inds.unsqueeze(1)
        logging.debug("temp_t shape: {}".format(temp_t.shape))

        # diff shape: [batch_size, seq_length, seq_length]
        # diff中为0的表示相同 即diff[k][i][j]==0表示第k样本中，位置i和位置j相同
        diff = temp - temp_t
        logging.debug("diff shape: {}".format(diff.shape))
        # 将diff中不为0的置零，为0的置1
        duplicate = torch.where(diff==0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        logging.debug("duplicate shape: {}".format(duplicate.shape))
        # 只保留上三角 且不留对角线
        # 因此当第k样本中位置i和位置j(i<j)相同时，只会有duplicate[k][i][j]=1
        # 而duplicate[k][j][i]在下三角 为0
        # 意为 重复时，只有后面出现的有损失
        duplicate = torch.triu(duplicate, diagonal=1)
        logging.debug("duplicate shape: {}".format(duplicate.shape))
        # 按照第二维聚合 也是意为将重复的损失算在后出现的位置上
        # 且为相加 意为多次重复的损失更重
        # duplicate_mask shape: [batch_size, seq_length]
        duplicate_mask = duplicate.sum(dim=1)
        logging.debug("duplicate_mask shape: {}".format(duplicate_mask.shape))
        #logging.info("duplicate_mask[:10]: {}".format(duplicate_mask[:10]))

        # 根据mask 只考虑重复的位置
        # duplicate_loss shape: [batch_size, seq_length]
        duplicate_loss = max_values * duplicate_mask
        logging.debug("duplicate_loss shape: {}".format(duplicate_loss.shape))

        # target_mask
        target_mask = token_type_ids[:, 1:]

        duplicate_loss = (duplicate_loss * target_mask).sum() / target_mask.sum()

        ## 同batch_size的duplicate_loss相加
        ## 跨batch_size的平均
        #duplicate_loss = duplicate_loss.sum(dim=1).mean()
        logging.debug("duplicate_loss shape: {}".format(duplicate_loss.shape))

        # 返回多的
        return duplicate_loss

    def train(self, *args, **kwargs):
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(BertSeq2seqModel, self).train(*args, **kwargs)

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        # 验证时不保存反向的梯度
        with torch.no_grad():
            for batch in eval_dataloader:
                cur_eval_step += 1
                loss = self.get_loss(*batch)
                # 保存loss时 先将其detach 不然保存的不只是loss 还有整个计算图
                loss_list.append(loss.detach().item())
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

    def gen_poem(self, beam_size=3, is_poem=True):
        test_data = ["北国风光##五言绝句", "题西林壁##七言绝句", "长安早春##五言律诗"]
        for text in test_data:
            logging.info(text)
            logging.info(self.generate(text, beam_size=beam_size, device=self.device, is_poem=is_poem))
