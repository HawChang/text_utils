#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   dygraph.py
Author:   zhanghao55@baidu.com
Date  :   20/08/14 16:26:10
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

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_cur_dir, "../../"))
from lib.common.data_io import gen_batch_data

def train(model, optimizer, train_data, eval_data, label_encoder,
          model_save_path=None, best_model_save_path=None,
          epochs=5, batch_size=32, max_seq_len=300,
          max_ensure=False, best_acc=0, print_step=50, logits_softmax=True):
    """ ѵ��dygraphģ��
    [IN]  model: dygraphģ�ͽṹ
          optimizer: �Ż���
          train_data: list[(input1[, input2, ...], label)], ѵ������
          eval_data: list[(input1[, input2, ...], label)], ��������
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
    [OUT] best_acc: float, ѵ���õ�������acc
    """
    logging.info("train model start")
    train_start_time = time.time()
    # ����trainģʽ
    model.train()
    cur_train_step = 0
    for cur_epoch in range(epochs):
        # ÿ��epoch��shuffle�����Ի�����ѵ��Ч����
        np.random.shuffle(train_data)
        train_data_batch = gen_batch_data(train_data, batch_size, max_seq_len, max_ensure)
        for cur_train_data, cur_train_label in train_data_batch:
            cur_train_step += 1
            # ѵ������תtensor
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            # ģ�͵ķ���ֵ����(loss, logits)������logitsĿǰ��ʱ����Ҫʹ��
            loss, _ = model(cur_train_data, labels=cur_train_label)
            # ���򴫲�
            loss.backward()
            optimizer.minimize(loss)
            # ����ݶ�
            model.clear_gradients()
            if cur_train_step % print_step == 0:
                speed = cur_train_step / (time.time() - train_start_time)
                logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), speed))

        # ������֤��׼ȷ��
        acc = eval(model, eval_data, label_encoder, batch_size=batch_size,
                max_seq_len=max_seq_len, logits_softmax=logits_softmax)
        logging.info('eval epoch %d, acc %.5f' % (cur_epoch, acc))

        if model_save_path is not None:
            # ÿ�ֱ���ģ��
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

            if best_model_save_path is not None and acc > best_acc:
                # �����������acc �򱣴�Ϊbestģ��
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc
    logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
    return best_acc


def infer(model, infer_data, max_seq_len=300, is_tensor=True, logits_softmax=True):
    """ ��dygraphģ��Ԥ��
    [IN]  model: dygraphģ�ͽṹ
          infer_data: list[(input1[, input2, ...])], ��Ԥ������
          max_seq_len: int, ��󳤶�
          is_tensor: boolean, true��infer_data�Ѿ���paddle�ɴ����tensor
          logits_softmax: boolean, true��Ԥ����Ϊsoftmax���logits
    [OUT] pred: list[float], Ԥ����
    """
    # �����with����ernie��������ݶȼ��㣻
    with D.base._switch_tracer_mode_guard_(is_train=False):
        # ����ģ�ͽ���evalģʽ���⽫��ر����е�dropout��
        model.eval()
        # ���infer_dataû��תtensor ��תΪpaddle���յ�tensor
        if not is_tensor:
            infer_data = D.to_variable(np.array(infer_data))

        logits = model(infer_data, logits_softmax=logits_softmax)
        # TODO: ����rateֵ
        pred = L.argmax(logits, -1).numpy()
        # ����trainģʽ
        model.train()
    return pred


def batch_infer(model, infer_data, batch_size=32, max_seq_len=300,
        print_step=20, logits_softmax=True):
    """ ��dygraphģ������Ԥ��
    [IN]  model: dygraphģ�ͽṹ
          infer_data: list[(input1[, input2, ...])], ��Ԥ������
          batch_size: int, ����С
          max_seq_len: int, ��󳤶�
          print_step: int, ÿ��print_step��ӡѵ�����
          logits_softmax: boolean, true��Ԥ����Ϊsoftmax���logits
    [OUT] pred: list[float], Ԥ����
    """
    all_pred = []
    all_label = []
    # inferdata������
    infer_data_batch = gen_batch_data(
            infer_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len)

    cur_infer_step = 0
    cur_infer_time = time.time()
    for cur_infer_data, cur_infer_label in infer_data_batch:
        cur_infer_data = D.to_variable(cur_infer_data)
        cur_infer_label = D.to_variable(cur_infer_label)
        cur_infer_step += 1
        cur_pred = infer(model, cur_infer_data, max_seq_len, True, logits_softmax)
        #cur_data = cur_infer_data.numpy()
        cur_label = np.squeeze(cur_infer_label.numpy(), axis=-1)
        all_pred.extend(cur_pred)
        all_label.extend(cur_label)
        if cur_infer_step % print_step == 0:
            cost_time = time.time() - cur_infer_time
            speed = cur_infer_step / cost_time
            logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                    % (cur_infer_step, cost_time, speed))

    return all_pred, all_label


def eval(model, eval_data, label_encoder, batch_size=32, max_seq_len=300,
        print_step=20, logits_softmax=True, report=True):
    """ ����dygraphģ��
    [IN]  model: dygraphģ�ͽṹ
          eval_data: list[(input1[, input2, ...], label)], ѵ������
          label_encoder: LabelEncoder, ���ת������
          batch_size: int, ����С
          max_seq_len: int, ��󳤶�
          print_step: int, ÿ��print_step��ӡѵ�����
          logits_softmax: boolean, true����֤ʱ���softmax���logits
          report: boolean, true��չʾclassification_report��Ϣ
    [OUT] acc: float, ������acc���
    """
    all_pred, all_label = batch_infer(model, eval_data, batch_size,
            max_seq_len, print_step, logits_softmax)

    all_pred = [label_encoder.inverse_transform(x) for x in all_pred]
    all_label = [label_encoder.inverse_transform(x) for x in all_label]
    if report:
        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
    acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
    return acc


def KL(pred, target, temperature=5, verbose=False):
    """ ����pred��target��KLɢ��
    [IN]  pred: numpy.ndarray, Ԥ��ֵ
          target: numpy.ndarray, Ŀ��ֵ
          temperature: int, �¶Ȳ���T
          verbose: boolean, true��չʾ����T, softmax���pred��target�����
    [OUT] loss: float, KLɢ�Ƚ��
    """
    #logging.info("pred before: {}".format(pred.numpy()[:2]))
    pred = L.softmax(pred / temperature)
    if verbose:
        logging.info("pred: {}".format(pred.numpy()[:2]))
    pred = L.log(pred)
    #logging.info("target before: {}".format(target.numpy()[:2]))
    target = L.softmax(target / temperature)
    if verbose:
        logging.info("target: {}".format(target.numpy()[:2]))
    loss = L.kldiv_loss(pred, target)
    if verbose:
        logging.info("kl loss: {}".format(loss.numpy()))
    return loss


def distill(model_t, model_s, optimizer, train_data, eval_data, label_encoder,
            unmark_data=None, model_save_path=None, best_model_save_path=None,
            epochs=5, batch_size=32, max_seq_len=300, best_acc=0,
            temperature=5, print_step=50):
    """ ��model_t����model_sģ��
    [IN]  model_t: teacherģ��
          model_s: studentģ��
          optimizer: �Ż���
          train_data: list[(input1[, input2, ...], label)], ѵ������
          eval_data: list[(input1[, input2, ...], label)], ��������
          label_encoder: LabelEncoder, ���ת������
          unmark_data: list[(input1[, input2, ...], )], δѵ������
          model_save_path: string, ģ�ʹ洢·��
          best_model_save_path: string, ����ģ�ʹ洢·��
          epochs:  int, ѵ������
          batch_size: int, ����С
          max_seq_len: int, ��󳤶�
          best_acc: float, ����acc��ʼֵ
          temperature: int, �¶Ȳ���T
          print_step: int, ÿ��print_step��ӡѵ�����
    [OUT] best_acc: float, ����õ�������acc
    """
    logging.info("distill model start")
    # teacherģ��תΪeval״̬
    model_t.eval()
    # studentģ��תΪtrain״̬
    model_s.train()

    # ѵ����Ϣͳ��
    distill_start_time = time.time()
    total_train_with_label_time = 0.0
    total_train_without_label_time = 0.0
    cur_train_step = 0
    cur_unmark_train_step = 0

    for cur_epoch in range(epochs):
        # ÿ��epoch��shuffle�����Ի�����ѵ��Ч����
        np.random.shuffle(train_data)
        # ��ñ�ע���ϵ�����
        train_data_batch = gen_batch_data(train_data, batch_size=batch_size, max_seq_len=max_seq_len)
        # ��ע����ѵ��
        for cur_train_data, cur_train_label in train_data_batch:
            cur_train_step += 1
            train_with_label_time_begin = time.time()
            # ����תΪtensor
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            # teacherģ�ͼ���ʱ����Ҫ�ݶȼ��㣻
            with D.base._switch_tracer_mode_guard_(is_train=False):
                # softmaxǰ��logits
                logits_t = model_t(cur_train_data, logits_softmax=False)
            # teacherģ�Ͳ����򴫲�
            logits_t.stop_gradient=True
            # studentģ�͵Ľ��
            loss_s, logits_s = model_s(cur_train_data, labels=cur_train_label)
            # ������ģ�������KL��ʧ
            loss_kl = KL(logits_s, logits_t, temperature=temperature)
            # �ϲ���ʧ
            loss = pow(temperature, 2) * loss_kl + loss_s
            # ���򴫲� �����ݶ�
            loss.backward()
            # ����
            optimizer.minimize(loss)
            # ��ձ����ݶ�
            model_s.clear_gradients()
            total_train_with_label_time += (time.time() - train_with_label_time_begin)
            if cur_train_step % print_step == 0:
                speed = cur_train_step / total_train_with_label_time
                logging.info('distill train epoch %d, step %d: loss %.5f(%.5f, %.5f), speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), loss_kl.numpy(), loss_s.numpy(), speed))

        # ���㵱ǰ����֤���ϵ�׼ȷ��
        acc = eval(model_s, eval_data, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
        logging.info('student eval epoch %d, acc %.5f' % (cur_epoch, acc))

        # ���������ʷacc ����Ϊbestģ��
        if best_model_save_path is not None and acc > best_acc:
            logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model_s.state_dict(), best_model_save_path)
            logging.info("cost time: %.4fs" % (time.time() - start_time))
            best_acc = acc

        if unmark_data is not None:
            unmark_train_data_batch = gen_batch_data(unmark_data,
                    batch_size=batch_size, max_seq_len=max_seq_len, with_label=False)
            # δ��ע����ѵ��
            for (cur_unmark_train_data,) in unmark_train_data_batch:
                cur_unmark_train_step += 1
                train_without_label_time_begin = time.time()
                # תΪpadddle����tensor
                cur_unmark_train_data = D.to_variable(cur_unmark_train_data)
                # teacherģ�ͼ���ʱ����Ҫ�ݶȼ��㣻
                with D.base._switch_tracer_mode_guard_(is_train=False):
                    logits_t = model_t(cur_unmark_train_data, logits_softmax=False)
                # teacherģ��ֹͣ���򴫲�
                logits_t.stop_gradient=True
                # studentģ�ͽ��
                logits_s = model_s(cur_unmark_train_data, logits_softmax=False)
                # ������ģ�������KL��ʧ
                loss_kl = KL(logits_s, logits_t, verbose=False)
                # ��ʧ�������
                loss = pow(temperature, 2) * loss_kl
                # ���򴫲�
                loss.backward()
                optimizer.minimize(loss)
                # ����ݶ�
                model_s.clear_gradients()
                total_train_without_label_time += (time.time() - train_without_label_time_begin)
                if cur_unmark_train_step % print_step == 0:
                    speed = cur_unmark_train_step / total_train_without_label_time
                    logging.info('distill unmark train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_unmark_train_step, loss.numpy(), speed))

            # ���㵱ǰ����֤���ϵ�׼ȷ��
            acc = eval(model_s, eval_data, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
            logging.info('student unmark eval epoch %d, acc %.5f' % (cur_epoch, acc))

            # ���������ʷacc ����Ϊbestģ��
            if best_model_save_path is not None and acc > best_acc:
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model_s.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc

        # ÿ��ѵ����󱣴�ģ��
        if model_save_path is not None:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model_s.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

    # teacherģ��תΪtrainģʽ
    model_t.train()
    logging.info("distill model cost time %.4fs" % (time.time() - distill_start_time))
    return best_acc


if __name__ == "__main__":
    pass
