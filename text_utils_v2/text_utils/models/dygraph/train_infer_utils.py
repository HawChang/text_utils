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

#_cur_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(_cur_dir, "../../"))
from text_utils.utils.data_io import gen_batch_data

def train(model, optimizer, train_data_list, eval_data_list, label_encoder,
        model_save_path=None, best_model_save_path=None,
        epochs=5, batch_size=32, max_seq_len=300,
        max_ensure=False, best_acc=0, print_step=50, logits_softmax=True):
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
    [OUT] best_acc: float, 训练得到的最优acc
    """
    logging.info("train model start")
    train_start_time = time.time()
    # 进入train模式
    model.train()
    cur_train_step = 0
    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果；
        np.random.shuffle(train_data_list)
        train_data_batch = gen_batch_data(train_data_list, batch_size, max_seq_len, max_ensure)
        for cur_train_data, cur_train_label in train_data_batch:
            cur_train_step += 1
            # 训练数据转tensor
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            # 模型的返回值包含(loss, logits)；其中logits目前暂时不需要使用
            loss, _ = model(cur_train_data, labels=cur_train_label)
            # 反向传播
            loss.backward()
            optimizer.minimize(loss)
            # 清空梯度
            model.clear_gradients()
            if cur_train_step % print_step == 0:
                speed = cur_train_step / (time.time() - train_start_time)
                logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), speed))

        # 计算验证集准确率
        acc = eval(model, eval_data_list, label_encoder, batch_size=batch_size,
                max_seq_len=max_seq_len, logits_softmax=logits_softmax)
        logging.info('eval epoch %d, acc %.5f' % (cur_epoch, acc))

        if model_save_path is not None:
            # 每轮保存模型
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

            if best_model_save_path is not None and acc > best_acc:
                # 如果优于最优acc 则保存为best模型
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc
    logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
    return best_acc


def infer(model, infer_data, max_seq_len=300, is_tensor=True, logits_softmax=True):
    """ 用dygraph模型预测
    [IN]  model: dygraph模型结构
          infer_data: list[(input1[, input2, ...])], 待预测数据
          max_seq_len: int, 最大长度
          is_tensor: boolean, true则infer_data已经是paddle可处理的tensor
          logits_softmax: boolean, true则预测结果为softmax后的logits
    [OUT] pred: list[float], 预测结果
    """
    # 在这个with域内ernie不会进行梯度计算；
    with D.base._switch_tracer_mode_guard_(is_train=False):
        # 控制模型进入eval模式，这将会关闭所有的dropout；
        model.eval()
        # 如果infer_data没有转tensor 则转为paddle接收的tensor
        if not is_tensor:
            infer_data = D.to_variable(np.array(infer_data))

        if logits_softmax is None:
            logits = model(infer_data)
        else:
            logits = model(infer_data, logits_softmax=logits_softmax)

        if isinstance(logits, tuple):
            # 如果是tuple 表示是多输出
            # tuple每个元素是每个输出的结果
            # 每个输出的结果都是整个batch的结果列表
            # 因此logits结果其实是按输出聚合的
            # 但我们需要按batch的样本聚合
            # 即需要将batch中各样本的不同输出组合到一起
            # 所有样本的结果最后组成列表
            # 以方便后期使用
            # 例：输入X1，对应输出(Y11，Y12)；输入X2，对应输出(Y21，Y22)
            # 输入[X1, X2]
            # 原logits=[[Y11, Y21], [Y12, Y22]], 即logits的下标对应的是各输出
            # 需要转成logits=[[Y11, Y12], [Y21, Y22]]，即logits的下标对应的是各样本
            output_list = list()
            logits_np_list = [x.numpy() for x in logits]
            # 转化前：logits_np_list长度为输出的个数，logits_np_list中各元素的一维长度为batch的大小
            for index, cur_logits_np in enumerate(logits_np_list):
                cur_np_list = [x for x in cur_logits_np]
                output_list.append(cur_np_list)
            # output_list是各输出结果的列表，要将各输出的结果zip到一起，需要*表示将output_list列表的每个元素zip到一起
            # 转化后：logits长度为batch大小, logits中各元素大小为输出的个数
            logits = zip(*output_list)
            #logging.info("len logits: {}".format(len(logits)))
        else:
            logits = logits.numpy()
        # 进入train模式
        model.train()
    return logits


def batch_infer(model, data_iter, with_label=True, batch_size=32, max_seq_len=300,
        print_step=20, logits_softmax=True):
    """ 用dygraph模型逐批预测
    [IN]  model: dygraph模型结构
          data_iter: iterable[(input1[, input2, ...])], 待预测数据
          with_label: boolean, true则data_iter中为(数据,标签)二元组列表
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
          print_step: int, 每个print_step打印训练情况
          logits_softmax: boolean, true则预测结果为softmax后的logits
    [OUT] pred: list[float], 预测结果
    """
    all_logits = []
    if with_label:
        all_label = []

    # infer data不打乱
    infer_data_batch = gen_batch_data(
            data_iter,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            with_label=with_label,
            )

    cur_infer_step = 0
    cur_infer_time = time.time()
    for cur_batch in infer_data_batch:
        if with_label:
            cur_infer_data, cur_infer_label = cur_batch
        else:
            cur_infer_data = cur_batch
        cur_infer_data = D.to_variable(cur_infer_data)
        cur_infer_step += 1
        cur_logits = infer(model, cur_infer_data, max_seq_len, True, logits_softmax)

        all_logits.extend(cur_logits)

        if with_label:
            cur_label = np.squeeze(cur_infer_label, axis=-1)
            all_label.extend(cur_label)

        if cur_infer_step % print_step == 0:
            cost_time = time.time() - cur_infer_time
            speed = cur_infer_step / cost_time
            logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                    % (cur_infer_step, cost_time, speed))

    if with_label:
        return all_logits, all_label
    else:
        return all_logits


def eval(model, eval_data, label_encoder, batch_size=32, max_seq_len=300,
        print_step=20, logits_softmax=True, report=True):
    """ 评估dygraph模型
    [IN]  model: dygraph模型结构
          eval_data: list[(input1[, input2, ...], label)], 训练数据
          label_encoder: LabelEncoder, 类别转化工具
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
          print_step: int, 每个print_step打印训练情况
          logits_softmax: boolean, true则验证时输出softmax后的logits
          report: boolean, true则展示classification_report信息
    [OUT] acc: float, 评估的acc结果
    """
    all_logits, all_label = batch_infer(model, eval_data, True, batch_size,
            max_seq_len, print_step, logits_softmax)

    all_pred = np.argmax(all_logits, axis=-1)
    all_pred = [label_encoder.inverse_transform(x) for x in all_pred]
    all_label = [label_encoder.inverse_transform(x) for x in  all_label]
    #all_pred = map(lambda x: label_encoder.inverse_transform(x), all_pred)
    #all_label = map(lambda x: label_encoder.inverse_transform(x), all_label)

    if report:
        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
    acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
    return acc


def KL(pred, target, temperature=5, verbose=False):
    """ 计算pred和target的KL散度
    [IN]  pred: numpy.ndarray, 预测值
          target: numpy.ndarray, 目标值
          temperature: int, 温度参数T
          verbose: boolean, true则展示除以T, softmax后的pred和target的输出
    [OUT] loss: float, KL散度结果
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


def distill(model_t, model_s, optimizer, train_data_list, eval_data_list, label_encoder,
            unmark_data=None, model_save_path=None, best_model_save_path=None,
            epochs=5, batch_size=32, max_seq_len=300, best_acc=0,
            temperature=5, print_step=50):
    """ 用model_t蒸馏model_s模型
    [IN]  model_t: teacher模型
          model_s: student模型
          optimizer: 优化器
          train_data_list: list[(input1[, input2, ...], label)], 训练数据
          eval_data_list: list[(input1[, input2, ...], label)], 评估数据
          label_encoder: LabelEncoder, 类别转化工具
          unmark_data: list[(input1[, input2, ...], )], 未训练数据
          model_save_path: string, 模型存储路径
          best_model_save_path: string, 最优模型存储路径
          epochs:  int, 训练轮数
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
          best_acc: float, 最优acc初始值
          temperature: int, 温度参数T
          print_step: int, 每个print_step打印训练情况
    [OUT] best_acc: float, 蒸馏得到的最优acc
    """
    logging.info("distill model start")
    # teacher模型进入eval即可
    model_t.eval()
    # student模型进入train模式
    model_s.train()

    # 训练信息统计
    distill_start_time = time.time()
    total_train_with_label_time = 0.0
    total_train_without_label_time = 0.0
    cur_train_step = 0
    cur_unmark_train_step = 0

    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果；
        np.random.shuffle(train_data_list)
        train_data_batch = gen_batch_data(train_data_list, batch_size=batch_size, max_seq_len=max_seq_len)
        # 标注数据训练
        for cur_train_data, cur_train_label in train_data_batch:
            cur_train_step += 1
            train_with_label_time_begin = time.time()
            # 输入转为tensor
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            # teacher模型计算时不需要梯度计算；
            with D.base._switch_tracer_mode_guard_(is_train=False):
                # softmax前的logits
                logits_t = model_t(cur_train_data, logits_softmax=False)
            # teacher模型停止反向传播
            logits_t.stop_gradient=True
            # student模型的结果
            loss_s, logits_s = model_s(cur_train_data, labels=cur_train_label, logits_softmax=False)
            # 计算两模型输出的KL损失
            loss_kl = KL(logits_s, logits_t, temperature=temperature)
            # 合并损失
            loss = pow(temperature, 2) * loss_kl + loss_s
            # 反向传播
            loss.backward()
            optimizer.minimize(loss)
            # 清空梯度
            model_s.clear_gradients()
            total_train_with_label_time += (time.time() - train_with_label_time_begin)
            if cur_train_step % print_step == 0:
                speed = cur_train_step / total_train_with_label_time
                logging.info('distill train epoch %d, step %d: loss %.5f(%.5f, %.5f), speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), loss_kl.numpy(), loss_s.numpy(), speed))

        # 计算当前在验证集上的准确率
        acc = eval(model_s, eval_data_list, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
        logging.info('student eval epoch %d, acc %.5f' % (cur_epoch, acc))

        # 如果优于历史acc 则保留为best模型
        if best_model_save_path is not None and acc > best_acc:
            logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model_s.state_dict(), best_model_save_path)
            logging.info("cost time: %.4fs" % (time.time() - start_time))
            best_acc = acc

        if unmark_data is not None:
            unmark_train_data_batch = gen_batch_data(unmark_data,
                    batch_size=batch_size, max_seq_len=max_seq_len, with_label=False)
            # 未标注数据训练
            for (cur_unmark_train_data,) in unmark_train_data_batch:
                cur_unmark_train_step += 1
                train_without_label_time_begin = time.time()
                # 转为padddle输入tensor
                cur_unmark_train_data = D.to_variable(cur_unmark_train_data)
                # teacher模型计算时不需要梯度计算；
                with D.base._switch_tracer_mode_guard_(is_train=False):
                    logits_t = model_t(cur_unmark_train_data, logits_softmax=False)
                # teacher模型停止反向传播
                logits_t.stop_gradient=True
                # student模型结果
                logits_s = model_s(cur_unmark_train_data, logits_softmax=False)
                # 计算两模型输出的KL损失
                loss_kl = KL(logits_s, logits_t, verbose=False)
                # 损失结果计算
                loss = pow(temperature, 2) * loss_kl
                # 反向传播
                loss.backward()
                optimizer.minimize(loss)
                # 清空梯度
                model_s.clear_gradients()
                total_train_without_label_time += (time.time() - train_without_label_time_begin)
                if cur_unmark_train_step % print_step == 0:
                    speed = cur_unmark_train_step / total_train_without_label_time
                    logging.info('distill unmark train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_unmark_train_step, loss.numpy(), speed))

            # 计算当前在验证集上的准确率
            acc = eval(model_s, eval_data_list, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
            logging.info('student unmark eval epoch %d, acc %.5f' % (cur_epoch, acc))

            # 如果优于历史acc 则保留为best模型
            if best_model_save_path is not None and acc > best_acc:
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model_s.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc

        # 每轮训练完后保存模型
        if model_save_path is not None:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model_s.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

    # teacher模型转为train模式
    model_t.train()
    logging.info("distill model cost time %.4fs" % (time.time() - distill_start_time))
    return best_acc

if __name__ == "__main__":
    pass
