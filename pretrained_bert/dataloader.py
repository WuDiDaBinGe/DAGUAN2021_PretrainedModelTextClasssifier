# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午4:24
# @Author  : WuDiDaBinGe
# @FileName: dataloader.py
# @Software: PyCharm
import linecache
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch.utils.data as Data
import random

from pre_config import PreDatasetConfig

# BERT有两个预训练的任务，分别是 掩码语言模型（MLM）任务 和 句子连贯性判定（NSP）任务

# https://blog.csdn.net/u011550545/article/details/90373264
# pytorch大数据加载
# 数据量大，无法一次读取到内存中
# 数据存储在csv或者文本文件中(每一行是一个sample，包括feature和label)
# 要求：每次读取一小块数据到内存；能够batch；能够shuffle

# 自定义BertDataset，继承torch.utils.data.Dataset，重写__init__(),__len__(),__getitem__()，增加initial()
# BertTokenizer，能够按照词表将文本划分成token序列， 并将各个token转换成id值。

class BertDataset(Dataset):
    def __init__(self, config, device):
        """
        file_path: the path to the dataset file
        nraws: each time put nraws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        file_raws = 0
        # get the count of all samples
        with open(config.train_path, 'r') as f:
            for _ in f:
                file_raws += 1
        self.file_path = config.train_path
        self.config = config
        self.file_raws = file_raws
        self.n_raws = config.n_raws
        self.shuffle = config.shuffle
        self.tokenizer = BertTokenizer(config.vocab_path)
        self.device = device

    def initial(self):
        self.f_input = open(self.file_path, 'r')
        self.samples = list()

        # put nraw samples into memory 将原始样本放入内存
        for _ in range(self.n_raws):
            data = self.f_input.readline()  # data contains the feature and label
            if data:
                self.samples.append(data)
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    # 返回数据集长度，方便tqdm显示进度条长度
    def __len__(self):
        return self.file_raws

    # 每次如何读数据
    def __getitem__(self, item):
        idx = self.index[0]
        data = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:
            # all the samples in the memory have been used, need to get the new samples
            for _ in range(self.n_raws):
                data = self.f_input.readline()  # data contains the feature and label
                if data:
                    self.samples.append(data)
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)

        # 处理文字
        doc_dict = eval(data)
        title = doc_dict['title']
        context = doc_dict['content']
        full_text = title + " " + context
        # 将文本字符串转换成token的序列
        tokenized = self.tokenizer(full_text, max_length=self.config.max_length, padding="max_length", truncation=True)
        # 需要张量化
        tokens = torch.tensor(tokenized['input_ids'])
        attention_mask = torch.tensor(tokenized['attention_mask'])
        # detach()函数可以返回一个完全相同的tensor,新的tensor开辟与旧的tensor共享内存，新的tensor会脱离计算图，不会牵扯梯度计算
        # clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
        labels = tokens.detach().clone()
        # 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数
        rand = torch.rand(tokens.shape)
        # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = (rand < .15) * (tokens != 101) * (tokens != 102) * (tokens != 0)
        selection = torch.where(mask_arr == 1)
        tokens[selection] = 103
        return tokens.to(self.device), attention_mask.to(self.device), labels.to(self.device)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PreDatasetConfig()
    train_dataset = BertDataset(config, device)
    for _ in range(10):
        train_dataset.initial()
        train_iter = Data.DataLoader(dataset=train_dataset, batch_size=32)
        for _, data in enumerate(train_iter):
            print(data)
