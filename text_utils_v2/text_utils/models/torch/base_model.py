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
            state_dict = torch.load(model_path)
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
        # ����trainģʽ
        self.model.train()
        # ��ʼ���Ż���
        self.optimizer = self.init_optimizer(learning_rate, **kwargs)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            for cur_train_batch in tqdm(train_dataloader):
                cur_train_step += 1
                loss = self.get_loss(cur_train_batch)

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

    def get_loss(self, batch):
        """ѵ��ʱ��εõ�loss
        """
        raise NotImplementedError

    def evaluate(self, eval_dataloader, **kwargs):
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


#class ClassificationModel(BaseModel):
    #def infer(self, infer_data_list, **kwargs):
    #    """ ��dygraphģ��Ԥ��
    #    [IN]  model: dygraphģ�ͽṹ
    #          infer_data_list: list[(input1[, input2, ...])], ��Ԥ������
    #    [OUT] pred: list[float], Ԥ����
    #    """
    #    # ���������Ƿ���תΪpaddle���յ�tensor
    #    is_tensor = kwargs.pop("is_tensor", True)

    #    # �����with����ernie��������ݶȼ��㣻
    #    with D.base._switch_tracer_mode_guard_(is_train=False):
    #        # ����ģ�ͽ���evalģʽ���⽫��ر����е�dropout��
    #        self.model.eval()
    #        # ���infer_data_listû��תtensor ��תΪpaddle���յ�tensor
    #        if not is_tensor:
    #            infer_data_list = [D.to_variable(np.array(x)) for x in infer_data_list]

    #        infer_res = self.model(*infer_data_list, **kwargs)

    #        # ��������ۺϽ��
    #        if isinstance(infer_res, tuple):
    #            infer_res = tuple([x.numpy() for x in infer_res])
    #        else:
    #            infer_res = infer_res.numpy()

    #        # ����trainģʽ
    #        self.model.train()
    #    return infer_res

    #def batch_infer(self, infer_data_iter, batch_size=32, max_seq_len=300, max_ensure=False,
    #        print_step=20, **kwargs):
    #    """ ��dygraphģ������Ԥ��
    #    [IN]  model: dygraphģ�ͽṹ
    #          infer_data_iter: iterable[(input1[, input2, ...])], ��Ԥ������
    #          with_label: boolean, true��infer_data_iter��Ϊ(����,��ǩ)��Ԫ���б�
    #          batch_size: int, ����С
    #          max_seq_len: int, ��󳤶�
    #          print_step: int, ÿ��print_step��ӡѵ�����
    #          logits_softmax: boolean, true��Ԥ����Ϊsoftmax���logits
    #    [OUT] pred: tuple(list[float]), Ԥ����
    #    """
    #    infer_res_list = None

    #    # infer data������
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
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer")
        super(Seq2seqModel, self).__init__(*args, **kwargs)
        self.word2ix = self.tokenizer._token_dict

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
            out_puts_ids = self.beam_search_poem(text, token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=device)
        else:
            logging.debug("common beam_search")
            out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=device)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
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

        sep_id = word2ix["[SEP]"]

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
                end_counts = (output_ids == sep_id).sum(1)  # ͳ�Ƴ��ֵ�end���
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

    #def generate(self, text, beam_size, is_poem=True):
    #    return self.get_model().generate(text, beam_size=beam_size, device=self.device, is_poem=is_poem)

    def gen_poem(self, beam_size=3, is_poem=True):
        test_data = ["�������##���Ծ���", "�����ֱ�##���Ծ���", "�����紺##������ʫ"]
        for text in test_data:
            logging.info(text)
            logging.info(self.generate(text, beam_size=beam_size, device=self.device, is_poem=is_poem))
