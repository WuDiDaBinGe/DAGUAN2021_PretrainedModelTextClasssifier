# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午8:41
# @Author  : WuDiDaBinGe
# @FileName: config.py
# @Software: PyCharm
import torch
import numpy as np

class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding='random'):
        self.model_name = 'TextRNN_Att'
        self.train_path = dataset + '/datagrand_2021_train.csv'  # 训练集
        self.dev_path = dataset + '/datagrand_2021_train.csv'  # 验证集
        self.test_path = dataset + '/datagrand_2021_test.csv'  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.second_num_classes = 35    # len(self.class_list)  # level 2 label
        self.first_num_classes = 10     # level 1 label
        self.n_vocab = 30355            # 词表大小，在运行时赋值
        self.num_epochs = 10            # epoch数
        self.batch_size = 128           # mini-batch大小
        self.pad_size = 350             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数