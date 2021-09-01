# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午8:22
# @Author  : WuDiDaBinGe
# @FileName: dataloader.py
# @Software: PyCharm
import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from config import Config


def load_data(dir_path, test=False):
    dataset = pd.read_csv(dir_path, sep=',')
    if test is False:
        # 处理层级标签
        dataset['1-label'] = dataset['label'].map(lambda a: int(a.split('-')[0]))
        dataset['2-label'] = dataset['label'].map(lambda a: int(a.split('-')[1]))
    return dataset


def spilt_dataset_pd(dataset, frac=0.2):
    train_data = dataset.sample(frac=1 - frac, random_state=0, axis=0)
    test_data = dataset[~dataset.index.isin(train_data.index)]
    return train_data, test_data


class MyDataset(Dataset):
    def __init__(self, config, dataset, device, test=False):
        super(MyDataset, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer(config.vocab_path)
        self.dataset = dataset
        self.id_arr = np.asarray(self.dataset.iloc[:, 0])
        self.text_arr = np.asarray(self.dataset.iloc[:, 1])
        self.test = test
        if self.test is False:
            self.first_label_arr = np.asarray(self.dataset.iloc[:, 3])
            self.second_label_arr = np.asarray(self.dataset.iloc[:, 4])
        self.device = device

    def __getitem__(self, item):
        id_ = self.id_arr[item]
        id_ = torch.tensor(id_).to(self.device)
        token_ids = self.text_arr[item]
        tokenized = self.tokenizer(token_ids, max_length=self.config.pad_size, truncation=True)
        token_ids = tokenized['input_ids']
        masks = tokenized['attention_mask']
        # padding and truncated
        padding_len = self.config.pad_size - len(token_ids)
        if padding_len >= 0:
            token_ids = token_ids + [0] * padding_len
            masks = masks + [0] * padding_len
        else:
            token_ids = token_ids[:self.config.pad_size]
            masks = masks[:self.config.pad_size]
        token_ids = torch.tensor(token_ids).to(self.device)
        masks = torch.tensor(masks, dtype=torch.bool).to(self.device)
        if self.test is False:
            first_label = self.first_label_arr[item]
            second_label = self.second_label_arr[item]
            first_label = torch.tensor(first_label - 1).to(self.device)
            second_label = torch.tensor(second_label - 1).to(self.device)
            return token_ids, masks, first_label, second_label
        else:
            return id_, token_ids, masks

    def __len__(self):
        return len(self.id_arr)


if __name__ == '__main__':
    config = Config(dataset='../dataset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_set = load_data(config.train_path)
    count = all_set.groupby(['2-label'], as_index=False)['2-label'].agg({'cnt': 'count'})
    loss = np.array(count)[:, 1]
    mean = np.mean(loss)
    loss = loss / mean
    print(loss)
    # train, dev = spilt_dataset_pd(all_set)
    # dataset = MyDataset(config=config, dataset=train, device=device)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # for iter, data in enumerate(dataloader):
    #     print(data)
