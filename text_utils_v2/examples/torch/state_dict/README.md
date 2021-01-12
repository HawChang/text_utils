# 预训练模型文件目录

## 文件目录格式
```bash
state_dict
├── bert_base_chinese
│   ├── bert-base-chinese-pytorch_model.bin
│   ├── bert-base-chinese-vocab.txt
│   ├── bert_config.json
│   ├── config.json -> bert_config.json
│   ├── pytorch_model.bin -> bert-base-chinese-pytorch_model.bin
│   └── vocab.txt -> bert-base-chinese-vocab.txt
├── ernie-1.0
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── README.md
└── roberta_wwm
    ├── config.json
    ├── pytorch_model.bin
    └── vocab.txt
```

## 模型数据结构

模型数据都由三个文件组成：
1. config.json：模型配置文件
2. pytorch_mode.bin：模型预训练参数
3. vocab.txt：模型词表

例如bert中，三个文件名与上述文件名不一致，可建立软链对应

## 数据地址

### 1. bert
1. bert-base-chinese-pytorch_model.bin在[这里](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin)
2. bert-base-chinese-vocab.txt在[这里](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt)

### 2. ernie 1.0
可以从paddle版的模型转，有现成的已转好的pytorch版预训练模型在[这里](https://github.com/nghuyong/ERNIE-Pytorch)

### 3. roberta
所需文件均在：[这里](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main)
