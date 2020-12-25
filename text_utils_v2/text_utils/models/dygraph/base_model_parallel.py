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

def model_parrallel(func):
    def wrapper(self, *args, **kwargs):
        logging.info("in wrapper")
        strategy = D.prepare_context()
        logging.info("strategy: {}".format(strategy))
        func(self, *args, **kwargs)
        self.model = D.DataParallel(self.model, strategy)
        logging.info("in wrapper")
    return wrapper


def gen_batch_data(data_iter, batch_size=32, max_seq_len=300, max_ensure=False):
    assert batch_size > 0, "batch_size should be greater than 0, actual= {}".format(batch_size)
    batch_data = list()

    def pad(data_list):
        """��data_list��Ԫ��pad���ȳ�
           ��Ԫ�ز���pad ����None
        """
        # ��������
        # ȷ����ǰdata_list�Ƿ����pad
        # ���б��Ԫ�ط��б��Ԫ��Ⱦ��г��ȵĶ���ʱ �޷�pad
        # ����None ��ʾpadʧ��
        cur_max_len = 0
        for cur_data in data_list:
            try:
                if len(cur_data) > cur_max_len:
                    cur_max_len = len(cur_data)
            except TypeError as e:
                return None

        # ���ָ������ ���滻Ϊ��󳤶�
        # �����Ϊָ����max_seq_len
        if max_ensure:
            cur_max_len = max_seq_len
        else:
            cur_max_len = max_seq_len if cur_max_len > max_seq_len else cur_max_len

        # padding
        return [np.pad(x[:cur_max_len], [0, cur_max_len-len(x[:cur_max_len])], mode='constant') for x in data_list]

    def batch_process(cur_batch_data, cur_batch_size):
        # cur_batch_dataΪ�б� �Ҳ�Ϊ��
        # Ԫ�ؾ�Ϊtuple
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
        # �涨:����ж������Ҫbatch ����zip��ϸ������б�
        # �������data_iter��Ԫ��Ϊtupleʱ��Ӧ����Ϊ������
        # �����벻�ǵ�ʱ����Ϊ�����룬���䴦��Ϊtuple
        if not isinstance(data, tuple):
            data = (data,)

        if len(batch_data) == batch_size:
            # ��ǰ�����һ��batch
            yield batch_process(batch_data, batch_size)
            batch_data = list()
        batch_data.append(data)

    if len(batch_data) > 0:
        yield batch_process(batch_data, len(batch_data))

class BaseModel(object):
    def __init__(self):
        """��ʼ��
        """
        self.built = False

    def init_optimizer(self, learning_rate):
        """��ʼ���Ż���
        """
        if not self.built:
            raise RuntimeError("model should be built before get optimizer")

        self.optimizer = F.optimizer.Adam(
                learning_rate=learning_rate,
                parameter_list=self.model.parameters())

    def save_model(self, save_path):
        """����ģ��
        """
        start_time = time.time()
        if D.parallel.Env().local_rank == 0:
            F.save_dygraph(self.model.state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path):
        """����ģ��
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
        logging.info("train model start")
        train_start_time = time.time()
        # ��������ģ��
        if load_best_model:
            self.load_model(best_model_save_path)
        # ����trainģʽ
        self.model.train()
        # ��ʼ���Ż���
        self.init_optimizer(learning_rate)

        cur_train_step = 0
        for cur_epoch in range(epochs):
            # ÿ��epoch��shuffle�����Ի�����ѵ��Ч����
            np.random.shuffle(train_data_list)
            def data_reader():
                return  gen_batch_data(train_data_list, batch_size*2, max_seq_len, max_ensure)
            # �����з�
            train_data_batch = F.contrib.reader.distributed_batch_reader(data_reader)()
            for cur_train_batch in train_data_batch:
                cur_train_step += 1
                cur_train_batch = [D.to_variable(x) for x in cur_train_batch]
                loss = self.get_loss(*cur_train_batch, **kwargs)
                #logging.info("loss type: {}".format(type(loss)))
                #logging.info("loss shape: {}".format(loss.shape))
                # ����ѵ����loss��һ��
                loss = self.model.scale_loss(loss)
                # ���򴫲�
                loss.backward()
                # ��ѵ�����ݶ��ռ�
                self.model.apply_collective_grads()

                self.optimizer.minimize(loss)
                # ����ݶ�
                self.model.clear_gradients()
                if cur_train_step % print_step == 0:
                    speed = cur_train_step / (time.time() - train_start_time)
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss.numpy(), speed))

            if model_save_path is not None:
                # ÿ�ֱ���ģ��
                logging.info("save model at epoch {}".format(cur_epoch))
                self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # ������֤��׼ȷ��
            cur_eval_res = self.evaluate(eval_data_list, batch_size=batch_size, max_seq_len=max_seq_len, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if best_model_save_path is not None and is_best:
                # ����ǵ�ǰ����Ч��ģ�� �򱣴�Ϊbestģ��
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def infer(self, infer_data_list, **kwargs):
        """ ��dygraphģ��Ԥ��
        [IN]  model: dygraphģ�ͽṹ
              infer_data_list: list[(input1[, input2, ...])], ��Ԥ������
        [OUT] pred: list[float], Ԥ����
        """
        # ���������Ƿ���תΪpaddle���յ�tensor
        is_tensor = kwargs.pop("is_tensor", True)

        # �����with����ernie��������ݶȼ��㣻
        with D.base._switch_tracer_mode_guard_(is_train=False):
            # ����ģ�ͽ���evalģʽ���⽫��ر����е�dropout��
            self.model.eval()
            # ���infer_data_listû��תtensor ��תΪpaddle���յ�tensor
            if not is_tensor:
                infer_data_list = [D.to_variable(np.array(x)) for x in infer_data_list]

            infer_res = self.model(*infer_data_list, **kwargs)

            # ��������ۺϽ��
            if isinstance(infer_res, tuple):
                infer_res = tuple([x.numpy() for x in infer_res])
            else:
                infer_res = infer_res.numpy()

            # ����trainģʽ
            self.model.train()
        return infer_res

    def batch_infer(self, infer_data_iter, batch_size=32, max_seq_len=300, max_ensure=False,
            print_step=20, **kwargs):
        """ ��dygraphģ������Ԥ��
        [IN]  model: dygraphģ�ͽṹ
              infer_data_iter: iterable[(input1[, input2, ...])], ��Ԥ������
              with_label: boolean, true��infer_data_iter��Ϊ(����,��ǩ)��Ԫ���б�
              batch_size: int, ����С
              max_seq_len: int, ��󳤶�
              print_step: int, ÿ��print_step��ӡѵ�����
              logits_softmax: boolean, true��Ԥ����Ϊsoftmax���logits
        [OUT] pred: tuple(list[float]), Ԥ����
        """
        infer_res_list = None

        # infer data������
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

    @model_parrallel
    def build(self, *args, **kwargs):
        """���繹������
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        """ģ��ѵ���׶ε���
        """
        raise NotImplementedError

    def evaluate(self, eval_data_list, batch_size=32, max_seq_len=300, **kwargs):
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
    def __init__(self):
        """��ʼ��
        """
        self.best_acc = None

    def get_loss(self, *input_list, **kwargs):
        input_label = input_list[-1]
        input_data = input_list[:-1]
        loss = self.model(*input_data, labels=input_label, **kwargs)
        # ģ�͵ķ���ֵ�����ɶ�� �涨��һ��Ϊloss
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
        """������������ж��Ƿ�����
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
        """��ʼ��
        """
        self.min_loss = None

    def get_loss(self, *input_list, **kwargs):
        loss = self.model(*input_list, **kwargs)
        # ģ�͵ķ���ֵ�����ɶ�� �涨��һ��Ϊloss
        if isinstance(loss, tuple):
            loss = loss[0]
        return loss

    def evaluate(self, eval_list, batch_size=32, max_seq_len=300, print_step=50, **kwargs):
        all_logits = self.batch_infer(eval_list,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                print_step=print_step,
                **kwargs)

        # loss������ĵ�һ��
        # �Ǹ��б�
        loss = np.mean(all_logits[0])
        logging.info("eval loss : {}".format(loss))
        return loss

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
