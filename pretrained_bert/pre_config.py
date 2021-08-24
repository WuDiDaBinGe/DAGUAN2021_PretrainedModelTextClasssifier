# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午5:10
# @Author  : WuDiDaBinGe
# @FileName: pre_config.py
# @Software: PyCharm
from transformers import BertConfig
class PreDatasetConfig(object):
    """配置参数"""

    def __init__(self):
        self.train_path = '/dataset/daguan_data_class/datagrand_2021_unlabeled_data.json'
        self.vocab_path = '../dataset/vocab.txt'
        # 数据集设置
        self.n_raws = 1000
        self.shuffle = False

        self.max_length = 512
        self.bert_config = BertConfig(
            vocab_size=30470,
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        self.lr = 1e-4
        self.num_epochs = 8
        self.batch_size = 16
