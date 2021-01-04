# 测试GRU在PY3下的问题

## 问题描述

1. test\_py3.py文件中的lstm网络，在python2&3下均可运行；但gru网络，在python2下可运行，在python3下会卡住，log日志停留在base\_model文件的166行("train model start")。
2. 如果把gru的网络结构调小，例如:`emb_dim=2`、`gru_dim=16`、`fc_hid_dim=32`，在python3下可运行

不清楚为什么会出现这种情况

## 问题复现

在README的同级目录运行test\_py3.py文件:`python test_py3.py`
