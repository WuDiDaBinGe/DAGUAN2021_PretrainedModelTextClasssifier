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

from pretrained_bert.pre_config import PreDatasetConfig


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

        # put nraw samples into memory
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

    def __len__(self):
        return self.file_raws

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
        tokenized = self.tokenizer(full_text, max_length=self.config.max_length, padding="max_length", truncation=True)
        tokens = torch.tensor(tokenized['input_ids'])
        attention_mask = torch.tensor(tokenized['attention_mask'])
        labels = tokens.detach().clone()
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
