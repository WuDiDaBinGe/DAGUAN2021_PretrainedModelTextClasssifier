# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午5:10
# @Author  : WuDiDaBinGe
# @FileName: pre_config.py
# @Software: PyCharm
from transformers import BertConfig
class PreDatasetConfig(object):
    """配置参数"""
    def __init__(self):
        self.train_path = '/home/wsj/dataset/2021达观杯/newdata.json'
        self.vocab_path = '/home/wsj/dataset/2021达观杯/vocab.txt'
        # 数据集设置
        self.n_raws = 1000
        self.shuffle = False

        self.max_length = 512
        # 用到了基于torch的transformers api，6层、12个注意力头、768维度已经512最大长度的序列。预训练的模式为MLM
        self.bert_config = BertConfig(
            vocab_size=30470,
            max_position_embeddings=512,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        self.lr = 1e-4
        self.num_epochs = 4
        self.batch_size = 16
