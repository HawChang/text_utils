#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   dygraph.py
Author:   zhanghao55@baidu.com
Date  :   20/08/14 16:26:10
Desc  :   
"""

import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
import time

from sklearn.metrics import classification_report

from data_io import gen_batch_data

def train(model, optimizer, train_data, eval_data, label_encoder, model_save_path=None,
        epochs=5, batch_size=32, max_seq_len=300, max_ensure=False, best_acc=0):
    logging.info("train model start")
    train_start_time = time.time()
    model.train()
    cur_train_step = 0
    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果；
        np.random.shuffle(train_data)
        train_data_batch = gen_batch_data(train_data, batch_size, max_seq_len, max_ensure)
        #epoch_start_time = time.time()
        for cur_train_data, cur_train_label in train_data_batch:
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            cur_train_step += 1
            # 模型的返回值包含(loss, logits)；其中logits目前暂时不需要使用
            loss, _ = model(cur_train_data, labels=cur_train_label)
            loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()
            actual_step = cur_train_step + 1
            if cur_train_step % 50 == 0:
                speed = cur_train_step / (time.time() - train_start_time)
                logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), speed))

        acc = eval(model, eval_data, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
        logging.info('eval epoch %d, acc %.5f' % (cur_epoch, acc))

        if model_save_path is not None:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

            if acc > best_acc:
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model.state_dict(), model_save_path + "_best")
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc
    logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
    return best_acc


def infer(model, infer_data, max_seq_len=300, is_tensor=True):
    # 在这个with域内ernie不会进行梯度计算；
    with D.base._switch_tracer_mode_guard_(is_train=False):
        # 控制模型进入eval模式，这将会关闭所有的dropout；
        model.eval()
        if not is_tensor:
            infer_data = D.to_variable(np.array(infer_data))
        _, logits = model(infer_data)
        #cur_data = cur_infer_data.numpy()
        pred = L.argmax(logits, -1).numpy()
        # 进入train模式
        model.train()
    return pred


def batch_infer(model, infer_data, batch_size=32, max_seq_len=300):
    all_pred = []
    all_label = []
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
        cur_pred = infer(model, cur_infer_data, max_seq_len, True)
        #cur_data = cur_infer_data.numpy()
        cur_label = np.squeeze(cur_infer_label.numpy(), axis=-1)
        all_pred.extend(cur_pred)
        all_label.extend(cur_label)
        if cur_infer_step % 20 == 0:
            cost_time = time.time() - cur_infer_time
            speed = cur_infer_step / cost_time
            logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' % (cur_infer_step, cost_time, speed))

    return all_pred, all_label


def eval(model, eval_data, label_encoder, batch_size=32, max_seq_len=300, report=True):
    all_pred, all_label = batch_infer(model, eval_data, batch_size, max_seq_len)

    all_pred = map(lambda x: label_encoder.inverse_transform(x), all_pred)
    all_label = map(lambda x: label_encoder.inverse_transform(x), all_label)
    if report:
        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
    acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
    return acc


def KL(pred, target, temperature=5, verbose=False):
    #logging.info("pred before: {}".format(pred.numpy()[:2]))
    pred = L.softmax(pred/temperature)
    if verbose:
        logging.info("pred: {}".format(pred.numpy()[:2]))
    pred = L.log(pred)
    #logging.info("target before: {}".format(target.numpy()[:2]))
    target = L.softmax(target/temperature)
    if verbose:
        logging.info("target: {}".format(target.numpy()[:2]))
    loss = L.kldiv_loss(pred, target)
    if verbose:
        logging.info("kl loss: {}".format(loss.numpy()))
    return loss


def distill(model_t, model_s, optimizer, train_data, eval_data, label_encoder,
            unmark_data=None, model_save_path=None, epochs=5, batch_size=32, max_seq_len=300,
            best_acc=0, kl_ratio=100, print_step=50):
    model_t.eval()
    model_s.train()
    logging.info("distill model start")
    distill_start_time = time.time()
    total_train_with_label_time = 0.0
    total_train_without_label_time = 0.0
    cur_train_step = 0
    cur_unmark_train_step = 0
    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果；
        np.random.shuffle(train_data)
        train_data_batch = gen_batch_data(train_data, batch_size=batch_size, max_seq_len=max_seq_len)
        # 标注数据蒸馏
        for cur_train_data, cur_train_label in train_data_batch:
            train_with_label_time_begin = time.time()
            cur_train_data = D.to_variable(cur_train_data)
            cur_train_label = D.to_variable(cur_train_label)
            cur_train_step += 1
            # 模型的返回值包含(loss, logits)；其中logits目前暂时不需要使用
            with D.base._switch_tracer_mode_guard_(is_train=False):
                _, logits_t = model_t(cur_train_data, labels=cur_train_label)
            logits_t.stop_gradient=True
            loss_s, logits_s = model_s(cur_train_data, labels=cur_train_label)
            loss_kl = KL(logits_s, logits_t)
            #logging.info("logits_s = {}".format(logits_s.numpy()[:2]))
            #logging.info("loss_s = {}".format(loss_s.numpy()))
            #logging.info("loss_kl = {}".format(loss_kl.numpy()))
            loss = kl_ratio * loss_kl + loss_s
            loss.backward()
            optimizer.minimize(loss)
            model_s.clear_gradients()
            total_train_with_label_time += (time.time() - train_with_label_time_begin)
            if cur_train_step % print_step == 0:
                speed = cur_train_step / total_train_with_label_time
                logging.info('distill train epoch %d, step %d: loss %.5f(%.5f, %.5f), speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), loss_kl.numpy(), loss_s.numpy(), speed))

        acc = eval(model_s, eval_data, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
        logging.info('student eval epoch %d, acc %.5f' % (cur_epoch, acc))

        if model_save_path is not None:
            if acc > best_acc:
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model_s.state_dict(), model_save_path + "_best")
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc

        if unmark_data is not None:
            unmark_train_data_batch = gen_batch_data(unmark_data, batch_size=batch_size, max_seq_len=max_seq_len, with_label=False)
            # 未标注数据蒸馏
            for (cur_unmark_train_data, ) in unmark_train_data_batch:
                train_without_label_time_begin = time.time()
                #print("unmark train data: {}".format(cur_unmark_train_data))
                cur_unmark_train_data = D.to_variable(cur_unmark_train_data)
                cur_unmark_train_step += 1
                # 模型的返回值包含(loss, logits)；其中logits目前暂时不需要使用
                with D.base._switch_tracer_mode_guard_(is_train=False):
                    _, logits_t = model_t(cur_unmark_train_data)
                logits_t.stop_gradient=True
                _, logits_s = model_s(cur_unmark_train_data)
                #verbose = True if cur_unmark_train_step % 500 == 0 else False
                verbose = False
                loss_kl = KL(logits_s, logits_t, verbose=verbose)
                loss = kl_ratio * loss_kl
                loss.backward()
                optimizer.minimize(loss)
                model_s.clear_gradients()
                total_train_without_label_time += (time.time() - train_without_label_time_begin)
                if cur_unmark_train_step % print_step == 0:
                    speed = cur_unmark_train_step / total_train_without_label_time
                    logging.info('distill unmark train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_unmark_train_step, loss.numpy(), speed))

            acc = eval(model_s, eval_data, label_encoder, batch_size=batch_size, max_seq_len=max_seq_len)
            logging.info('student unmark eval epoch %d, acc %.5f' % (cur_epoch, acc))

            if acc > best_acc:
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                F.save_dygraph(model_s.state_dict(), model_save_path + "_best")
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_acc = acc

        if model_save_path is not None:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            F.save_dygraph(model_s.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

    model_t.train()
    logging.info("distill model cost time %.4fs" % (time.time() - distill_start_time))
    return best_acc

if __name__ == "__main__":
    pass


