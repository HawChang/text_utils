#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   multi_label_metrics.py
Author:   zhanghao55@baidu.com
Date  :   20/07/08 15:33:58
Desc  :   
"""

import codecs
import json
import numpy as np

from sklearn.metrics import classification_report


def multilabel_classification_report(pred_labels, real_labels, label_encoder=None):
    """多标签
    """
    if isinstance(pred_labels, list) or isinstance(pred_labels, tuple):
        pred_labels = np.array(pred_labels)
    if isinstance(real_labels, list) or isinstance(real_labels, tuple):
        real_labels = np.array(real_labels)
    print("real labels: {}".format(pred_labels))
    print("real labels: {}".format(type(pred_labels)))
    pred_labels = pred_labels.astype(np.int32)
    real_labels = real_labels.astype(np.int32)
    assert pred_labels.shape == real_labels.shape, \
            "pred_labels' shape({}) != real_labels' shape({})".format(pred_labels.shape, real_labels.shape)
    # 总体预测数和类别数
    sample_num, label_num = pred_labels.shape


    report_tag = ["micro avg", "macro avg", "weighted avg"]
    metrics_list = ["", "precision", "recall", "f1-score", "support"]

    def display_report_dict(report_dict):
        """将dict打印
        """
        labels = sorted([x for x in report_dict.keys() if x not in report_tag])
        row_names = [""] + labels + [""] + report_tag

        def to_str(variable):
            """转为字符串
            """
            if isinstance(variable, float):
                return "%.2f" % variable

            return str(variable)

        # 确定每列的宽度
        width_dict = dict()
        for index, metrics in enumerate(metrics_list):
            width_dict[index] = 0
            for row_name in row_names:
                if metrics == "":
                    cur_value = row_name
                elif row_name == "":
                    cur_value = metrics
                else:
                    cur_value = report_dict[row_name][metrics]

                cur_value = to_str(cur_value)

                if len(cur_value) > width_dict[index]:
                    width_dict[index] = len(cur_value)

                #print("cur metrics: {}, cur row_name: {}, cur_value: {}, cur_length: {}" \
                #        .format(metrics, row_name, cur_value, width_dict[index]))

        def format_line(text_list, padding=4):
            res = ""
            for index, text in enumerate(text_list):
                res += text.rjust(width_dict[index] + padding)
            return res

        # 输出
        # 输出标题
        print(format_line(metrics_list))
        for row_name in row_names:
            # row_name为空则输出空行
            if row_name == "":
                print("")
            else:
                # 否则输出信息
                cur_info_list = list()
                for metrics in metrics_list:
                    if metrics == "":
                        cur_info = row_name
                    else:
                        cur_info = report_dict[row_name][metrics]
                    cur_info = to_str(cur_info)
                    cur_info_list.append(cur_info)
                print(format_line(cur_info_list))

    # 各类准确率情况
    res_list = list()
    # 遍历每个类别的预测情况
    for label_id in range(label_num):
        cur_label = label_encoder.inverse_transform(label_id)
        cur_report_dict = classification_report(real_labels[:,label_id], pred_labels[:,label_id], output_dict=True)
        print("label {}:".format(cur_label.encode("gb18030")))
        display_report_dict(cur_report_dict)
        res_list.append((cur_label, cur_report_dict["weighted avg"]))

    # 比较各结果
    match_matrix = np.isclose(pred_labels, real_labels)
    #print("match_matrix: \n{}".format(match_matrix))

    # 各类的正确数
    #correct_num_each_label = np.sum(match_matrix, axis=0)
    #print("correct_num_each_label: \n{}".format(correct_num_each_label))

    # 所有类都正确的数目
    correct_num_all_label = np.sum(np.all(match_matrix, axis=1))
    #print("correct_num_all_label: \n{}".format(correct_num_all_label))

    # acc each label
    #acc_each_label = correct_num_each_label / float(sample_num)

    # acc all label
    acc_all_label = correct_num_all_label / float(sample_num)
    print("total acc: {:.4f}%".format(acc_all_label * 100))

    return acc_all_label, res_list

def multilabel_classification_diff(pred_labels, real_labels, input_texts, label_encoder, output_path):
    if isinstance(pred_labels, list) or isinstance(pred_labels, tuple):
        pred_labels = np.array(pred_labels)
    if isinstance(real_labels, list) or isinstance(real_labels, tuple):
        real_labels = np.array(real_labels)
    pred_labels = pred_labels.astype(np.int32)
    real_labels = real_labels.astype(np.int32)
    with codecs.open(output_path, "w", "gb18030") as wf:
        for pred_label, real_label, text in zip(pred_labels, real_labels, input_texts):
            #print("real_label: {}".format(real_label))
            #print("pred_label: {}".format(pred_label))
            match_label = np.logical_xor(pred_label, real_label)
            #print("match_label: {}".format(match_label))
            diff_label = ""
            if match_label.any():
                # 如果有True 则有类别不一致
                diff_label_index = np.argwhere(match_label).squeeze(axis=-1)
                #print("diff label index: {}".format(diff_label_index))
                #print("diff label index shape: {}".format(diff_label_index.shape))
                #print("diff label index: {}".format(type(diff_label_index)))
                diff_label = map(lambda x: label_encoder.inverse_transform(x), diff_label_index)
                #print("diff label: {}".format(diff_label))
                diff_label = "|".join(diff_label)

            wf.write("\t".join([
                diff_label,
                json.dumps(pred_label.tolist()),
                json.dumps(real_label.tolist()),
                text,
                ]) + "\n")

if __name__ == "__main__":
    pass


