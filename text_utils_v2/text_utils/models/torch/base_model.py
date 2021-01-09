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
        """��ʼ��
        """
        # �ֲ�ʽѵ��ʱΪTrue
        self.distributed = False
        # ���ֲ�ʽѵ��ʱ local_rankΪ������ΨһID Ϊ0��Ϊ������
        # ����������ѵ��ʱ local_rankΪ0
        self.local_rank = 0
        # ���ֲ�ʽѵ�� ���ý��̲���������ʱ is_masterΪFalse�����������ΪTrue
        self.is_master = True
        self.model = self.init_model(*args, **kwargs)
        if not self.distributed:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def init_optimizer(self, learning_rate, weight_decay=1e-3, *args, **kwargs):
        """��ʼ���Ż���
        """
        return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

    def save_model(self, save_path):
        """����ģ��
        """
        start_time = time.time()
        torch.save(self.get_model().state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path, strict=True):
        """����ģ��
        """
        if os.path.exists(model_path):
            logging.info("load model from {}".format(model_path))
            start_time = time.time()
            # ��cpu�ϼ������� Ȼ����ص�ģ��
            # ��Ȼ�ڷֲ�ʽѵ��ʱ ����������cuda:0�ϼ���һ������
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            logging.debug("state_dict_names: {}".format(state_dict.keys()))
            self.get_model().load_state_dict(state_dict, strict=strict)
            torch.cuda.empty_cache()
            logging.info("cost time: %.4fs" % (time.time() - start_time))
        else:
            logging.info("cannot find model file: {}".format(model_path))

    def get_model(self):
        """ȡ��ģ��
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
        """ ѵ��dygraphģ��
        [IN]
              train_dataloader: DataLoader, ѵ������
              epochs:  int, ѵ������
              learning_rate: float, ѧϰ��
              print_step: int, ÿ��print_step��ӡѵ�����
              load_best_model: bool
        [OUT] best_acc: float, ѵ���õ�������acc
        """
        logging.info("trail train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()
        # ��������ģ��
        if load_best_model:
            self.load_model(best_model_save_path, strict)
        # ��ʼ���Ż���
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        trial_batch = next(iter(train_dataloader))
        cur_train_step = 0
        for cur_epoch in range(epochs):
            # ����trainģʽ
            # ÿepoch��Ҫtrain ��Ϊevaluate��ʱ����eval
            self.model.train()
            for _ in tqdm(range(print_step)):
                cur_train_step += 1
                loss = self.get_loss(*trial_batch)

                # ���֮ǰ���ݶ�
                self.optimizer.zero_grad()
                # ���򴫲�, ��ȡ�µ��ݶ�
                loss.backward()
                # �û�ȡ���ݶȸ���ģ�Ͳ���
                self.optimizer.step()

                if cur_train_step % print_step == 0:
                    speed = cur_train_step / (time.time() - train_start_time)
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss.cpu().detach().numpy(), speed))

            if self.is_master and model_save_path is not None:
                # ÿ�ֱ���ģ��
                logging.info("save model at epoch {}".format(cur_epoch))
                self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # ������֤��׼ȷ��
            cur_eval_res = self.evaluate([trial_batch], **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and best_model_save_path is not None and is_best:
                # ����ǵ�ǰ����Ч��ģ�� �򱣴�Ϊbestģ��
                logging.warning("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def train(self, train_dataloader, eval_dataloader,
            model_save_path=None, best_model_save_path=None,
            epochs=5, learning_rate=5e-5,
            print_step=50, load_best_model=True,
            strict=True, **kwargs):
        """ ѵ��dygraphģ��
        [IN]  model: dygraphģ�ͽṹ
              optimizer: �Ż���
              train_data_list: list[(input1[, input2, ...], label)], ѵ������
              eval_data_list: list[(input1[, input2, ...], label)], ��������
              label_encoder: LabelEncoder, ���ת������
              model_save_path: string, ģ�ʹ洢·��
              best_model_save_path: string, ����ģ�ʹ洢·��
              epochs:  int, ѵ������
              batch_size: int, ����С
              max_seq_len: int, ��󳤶�
              max_ensure: boolean, true��ʼ�ղ��뵽max_seq_len
              best_acc: float, ����acc��ʼֵ
              print_step: int, ÿ��print_step��ӡѵ�����
              logits_softmax: boolean, true����֤ʱ���softmax���logits
              eval_method: str, evalģ��Ч��
              with_label: boolean, true����������label
        [OUT] best_acc: float, ѵ���õ�������acc
        """
        logging.info("train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()
        # ��������ģ��
        if load_best_model:
            self.load_model(best_model_save_path, strict)
        # ��ʼ���Ż���
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            # ����trainģʽ
            # ÿepoch��Ҫtrain ��Ϊevaluate��ʱ����eval
            self.model.train()
            for cur_train_batch in tqdm(train_dataloader):
                cur_train_step += 1
                loss = self.get_loss(*cur_train_batch)

                #cur_train_batch = [x.to(self.device) for x in cur_train_batch]
                #loss = self.get_loss(*cur_train_batch, **kwargs)

                # ���֮ǰ���ݶ�
                self.optimizer.zero_grad()
                # ���򴫲�, ��ȡ�µ��ݶ�
                loss.backward()
                # �û�ȡ���ݶȸ���ģ�Ͳ���
                self.optimizer.step()

                if cur_train_step % print_step == 0:
                    speed = cur_train_step / (time.time() - train_start_time)
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss.cpu().detach().numpy(), speed))

            if self.is_master and model_save_path is not None:
                # ÿ�ֱ���ģ��
                logging.info("save model at epoch {}".format(cur_epoch))
                self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # ������֤��׼ȷ��
            cur_eval_res = self.evaluate(eval_dataloader, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and best_model_save_path is not None and is_best:
                # ����ǵ�ǰ����Ч��ģ�� �򱣴�Ϊbestģ��
                logging.warning("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def init_model(self, *args, **kwargs):
        """���繹������
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        """ѵ��ʱ��εõ�loss
        """
        raise NotImplementedError

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        """ģ������
        """
        raise NotImplementedError

    def check_if_best(self, cur_eval_res):
        """����������� �ж��Ƿ�����
        """
        raise NotImplementedError

    def get_best_score(self):
        """
        """
        raise NotImplementedError


class ClassificationModel(BaseModel):
    def __init__(self, best_acc=None, label_encoder=None, *args, **kwargs):
        """��ʼ��
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
        # ģ�͵ķ���ֵ�����ɶ�� �涨��һ��Ϊloss
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
        #    # ���ֲ�ʽѵ��ʱ ���Ҫ����ȫ����loss
        #    # �����²���
        #    #if gather_loss:
        #    #    loss_tensor = torch.tensor(loss_mean).to(self.device)
        #    #    # ����ֻ��ӡmaster���̵�loss ����ֻ��Ҫreduce��rankΪ0�Ľ���
        #    #    # ���Ҫ���н���loss_tensorͬ�� ��all_reduce
        #    #    torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
        #    #    if self.is_master:
        #    #        logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
        #    #        loss_mean = loss_tensor / torch.distributed.get_world_size()
        #    #        logging.infer("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
        #    #elif self.is_master:
        #    #    # ����ֻ��master���̴�ӡloss
        #    #    logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        #    logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        #else:
        #    logging.info("eval loss = {}.".format(loss_mean))

        #return loss_mean

    def check_if_best(self, cur_eval_res):
        """������������ж��Ƿ�����
        """
        if self.best_acc is None or self.best_acc <= cur_eval_res:
            self.best_acc = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        return self.best_acc

    def infer(self, *infer_data_list, **kwargs):
        """ ��dygraphģ��Ԥ��
        [IN]  model: dygraphģ�ͽṹ
              infer_data: list[(input1[, input2, ...])], ��Ԥ������
        [OUT] pred: list[float], Ԥ����
        """
        # ���������Ƿ���תΪpaddle���յ�tensor
        is_tensor = kwargs.pop("is_tensor", True)

        # inferʱ�����淴����ݶ�
        with torch.no_grad():
            # ����ģ�ͽ���evalģʽ���⽫��ر����е�dropout��norm��
            self.model.eval()
            # ���infer_data_listû��תtensor ��תΪtorch���յ�tensor
            if not is_tensor:
                infer_data_list = [torch.tensor(x, device=self.device) for x in infer_data_list]
            else:
                infer_data_list = [x.to(self.device) for x in infer_data_list]

            infer_res = self.model(*infer_data_list, **kwargs)

            # ��������ۺϽ��
            if isinstance(infer_res, tuple):
                infer_res = tuple([x.detach().cpu().numpy() for x in infer_res])
            else:
                infer_res = infer_res.detach().cpu().numpy()

        return infer_res

    def batch_infer(self, infer_dataloader, print_step=20, **kwargs):
        """ ��dygraphģ������Ԥ��
        [IN]  model: dygraphģ�ͽṹ
              infer_dataloader: DataLoader, ��Ԥ������
              print_step: int, ÿ��print_step��ӡѵ�����
              logits_softmax: boolean, true��Ԥ����Ϊsoftmax���logits
        [OUT] pred: tuple(list[float]), Ԥ����
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
        # �� һ�� ����������Ӧ�Ľ��
        ## ͨ�������󳤶ȵõ��������󳤶ȣ��������ⲻ�����������󳤶Ȼ���нض�
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length

        # print(text)
        # token_type_id ȫΪ0
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=device).view(1, -1)
        if is_poem:## ��ʫ��beam-search���в�ͬ
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
        beam-search����
        """
        yayun_pos = []
        title = text.split("##")[0]
        if "������ʫ" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "���Ծ���" in text:
            yayun_pos = [10, 22]
        elif "������ʫ" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "���Ծ���" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["��"]# ����
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["��"]# ���
        repeat_word = [[] for i in range(beam_size)]
        # ���������������
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
                    # �ظ�beam-size�� ����ids
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
    
    
                logit_score = output_scores.view(-1, 1) + logit_score # �ۼƵ÷�
                ## ȡtopk��ʱ��������չƽ��Ȼ����ȥ����topk����
                # չƽ
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # ������
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # ������
    
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
                        word = ix2word[last_chars[index].item()]# �ҵ���һ���ַ� ��ס��Ѻ�����
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
    
                # ���µ÷�
                output_scores = hype_score
    
                last_chars = indice2
    
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
    
                end_counts = (output_ids == sep_id).sum(1)  # ͳ�Ƴ��ֵ�end���
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # ˵��������ֹ�ˡ�
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # ����δ��ɲ���
                    flag = (end_counts < 1)  # ���δ�������
                    if not flag.all():  # ���������ɵ�
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # �ӵ����������
                        output_scores = output_scores[flag]  # �ӵ����������
                        end_counts = end_counts[flag]  # �ӵ������end����
                        beam_size = flag.sum()  # topk��Ӧ�仯
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
        beam-search����
        """
        # һ��ֻ����һ��
        # batch_size = 1

        # token_ids shape: [batch_size, seq_length]
        logging.debug("token_ids: {}".format(token_ids))
        logging.debug("token_ids shape: {}".format(token_ids.shape))

        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids : {}".format(token_type_ids))
        logging.debug("token_type_ids  shape: {}".format(token_type_ids.shape))

        #sep_id = word2ix["[SEP]"]

        # ���������������
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        logging.debug("output_ids: {}".format(output_ids))
        logging.debug("output_ids shape: {}".format(output_ids.shape))
        # ���������ۼƵ÷�

        with torch.no_grad():
            # ��ʼ�����÷�
            # output_scores shape: [batch_size]
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            # �ظ����� ֱ���ﵽ��󳤶�
            for step in range(self.out_max_length):
                logging.debug("beam size: {}".format(beam_size))
                if step == 0:
                    # score shape: [batch_size, seq_length, self.vocab_size]
                    scores = self.model(token_ids, token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                    # �ظ�beam-size�� ����ids
                    # token_ids shape: [beam_size, batch_size*seq_length]
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_ids shape: {}".format(token_ids.shape))

                    # token_type_ids shape: [beam_size, batch_size*seq_length]
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
                else:
                    # TODO score shape: [beam_size, cur_seq_length, self.vocab_size]
                    # cur_seq_length���𽥱仯��
                    scores = self.model(new_input_ids, new_token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                # ֻȡ���һ�������vocab�ϵ�score
                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = output_scores.view(-1, 1) + logit_score # �ۼƵ÷�
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                ## ȡtopk��ʱ��������չƽ��Ȼ����ȥ����topk����
                # չƽ
                # ����beam_size�ֽ����vocab�Ľ����ƽ
                logit_score = logit_score.view(-1)
                # �ҵ�topk��ֵ��λ��
                hype_score, hype_pos = torch.topk(logit_score, beam_size)

                # ���ݴ�ƽ���λ�� �ҵ����ƽǰ������λ��
                # ��λ����ʵ��beam_size�� �ĵڼ���beam ��λ�ÿ������ظ�
                # ��λ����ʵ�ǵ�ǰbeam�µ�vocab_id vocab_idҲ�������ظ�
                indice1 = (hype_pos // scores.shape[-1]) # ������
                logging.debug("indice1: {}".format(indice1))
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # ������
                logging.debug("indice2: {}".format(indice2))

                # ���µ÷�
                # output_scores shape: [beam_size]
                output_scores = hype_score
                logging.debug("output_scores: {}".format(output_scores))

                # ����output_ids
                # ͨ��indice1ѡ���ĸ�beam
                # ͨ��indice2ѡ��ǰbeam���ĸ�vocab
                # output_ids shape: [beam_size, cur_seq_length]
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                logging.debug("output_ids: {}".format(output_ids))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # token_ids�ǹ̶�ԭ����
                # output_ids�ǵ�ǰbeam_search���µ�beam_size����ѡ·��
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # output_ids��typeȫΪ1
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))

                # ��¼��ǰoutput_ids����sep_id�����
                end_counts = (output_ids == stop_id).sum(1)  # ͳ�Ƴ��ֵ�end���
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # �����ǰ�������ŵ��ѽ��� ��ѡ��ǰ��beam
                    # ˵��������ֹ�ˡ�
                    return output_ids[best_one]
                else :
                    # ���� ȥ������sep_id������
                    flag = (end_counts < 1)  # ���δ�������
                    if not flag.all():  # ���������ɵ�
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # �ӵ����������
                        output_scores = output_scores[flag]  # �ӵ����������
                        end_counts = end_counts[flag]  # �ӵ������end����
                        beam_size = flag.sum()  # topk��Ӧ�仯
                        logging.debug("beam size change")

            return output_ids[output_scores.argmax()]


class BertSeq2seqModel(Seq2seqModel):
    def __init__(self, *args, **kwargs):
        """��ʼ��
        """
        super(BertSeq2seqModel, self).__init__(*args, **kwargs)
        self.min_loss = None

    def get_loss(self, *batch):
        token_ids, token_type_ids, target_ids = [x.to(self.device) for x in batch]
        predictions, loss = self.model(token_ids,
                token_type_ids,
                labels=target_ids,
                device=self.device)

        ## predictionsȥ���˶����һ��sep��Ԥ����
        ## predictionsÿ��λ�ö��Ƕ�token_ids���¸�λ�õ�Ԥ��
        ## ������ĳһ��Ԥ��prediction[i]�����ӦԤ�����token_id[i+1]
        ## ���������Ҫ����prediction�ķ�Χ����ʵ��token_type_ids����һλ���õ�pred_mask
        ## ��ʱpred_mask��Ϊ1������Ҫ���ǵ�Ԥ����
        #pred_mask = token_type_ids[:, 1:].contiguous()

        ## ����Ҫ����ʫ��ʱ�� ���ظ����ɵ�id���гͷ�
        # TODO ʵ��Ч�������� ÿ��ʫ�����ظ����εı��
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
        # diff��Ϊ0�ı�ʾ��ͬ ��diff[k][i][j]==0��ʾ��k�����У�λ��i��λ��j��ͬ
        diff = temp - temp_t
        logging.debug("diff shape: {}".format(diff.shape))
        # ��diff�в�Ϊ0�����㣬Ϊ0����1
        duplicate = torch.where(diff==0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        logging.debug("duplicate shape: {}".format(duplicate.shape))
        # ֻ���������� �Ҳ����Խ���
        # ��˵���k������λ��i��λ��j(i<j)��ͬʱ��ֻ����duplicate[k][i][j]=1
        # ��duplicate[k][j][i]�������� Ϊ0
        # ��Ϊ �ظ�ʱ��ֻ�к�����ֵ�����ʧ
        duplicate = torch.triu(duplicate, diagonal=1)
        logging.debug("duplicate shape: {}".format(duplicate.shape))
        # ���յڶ�ά�ۺ� Ҳ����Ϊ���ظ�����ʧ���ں���ֵ�λ����
        # ��Ϊ��� ��Ϊ����ظ�����ʧ����
        # duplicate_mask shape: [batch_size, seq_length]
        duplicate_mask = duplicate.sum(dim=1)
        logging.debug("duplicate_mask shape: {}".format(duplicate_mask.shape))
        #logging.info("duplicate_mask[:10]: {}".format(duplicate_mask[:10]))

        # ����mask ֻ�����ظ���λ��
        # duplicate_loss shape: [batch_size, seq_length]
        duplicate_loss = max_values * duplicate_mask
        logging.debug("duplicate_loss shape: {}".format(duplicate_loss.shape))

        # target_mask
        target_mask = token_type_ids[:, 1:]

        duplicate_loss = (duplicate_loss * target_mask).sum() / target_mask.sum()

        ## ͬbatch_size��duplicate_loss���
        ## ��batch_size��ƽ��
        #duplicate_loss = duplicate_loss.sum(dim=1).mean()
        logging.debug("duplicate_loss shape: {}".format(duplicate_loss.shape))

        # ���ض��
        return duplicate_loss

    def train(self, *args, **kwargs):
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(BertSeq2seqModel, self).train(*args, **kwargs)

    def evaluate(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        # ��֤ʱ�����淴����ݶ�
        with torch.no_grad():
            for batch in eval_dataloader:
                cur_eval_step += 1
                loss = self.get_loss(*batch)
                # ����lossʱ �Ƚ���detach ��Ȼ����Ĳ�ֻ��loss ������������ͼ
                loss_list.append(loss.detach().item())
                if cur_eval_step % print_step == 0:
                    cost_time = time.time() - start_time
                    speed = cur_eval_step / cost_time
                    logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                            % (cur_eval_step, cost_time, speed))
        loss_mean = np.mean(loss_list)
        if self.distributed:
            # ���ֲ�ʽѵ��ʱ ���Ҫ����ȫ����loss
            # �����²���
            if gather_loss:
                loss_tensor = torch.tensor(loss_mean).to(self.device)
                # ����ֻ��ӡmaster���̵�loss ����ֻ��Ҫreduce��rankΪ0�Ľ���
                # ���Ҫ���н���loss_tensorͬ�� ��all_reduce
                torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
                if self.is_master:
                    logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
                    loss_mean = loss_tensor / torch.distributed.get_world_size()
                    logging.infer("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
            elif self.is_master:
                # ����ֻ��master���̴�ӡloss
                logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        else:
            logging.info("eval loss = {}.".format(loss_mean))

        if self.is_master:
            self.gen_poem(is_poem=False)

        return loss_mean

    def check_if_best(self, cur_eval_res):
        """������������ж��Ƿ�����
        """
        if self.min_loss is None or self.min_loss >= cur_eval_res:
            self.min_loss = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        return self.min_loss

    def gen_poem(self, beam_size=3, is_poem=True):
        test_data = ["�������##���Ծ���", "�����ֱ�##���Ծ���", "�����紺##������ʫ"]
        for text in test_data:
            logging.info(text)
            logging.info(self.generate(text, beam_size=beam_size, device=self.device, is_poem=is_poem))
