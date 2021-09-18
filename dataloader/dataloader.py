# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午8:22
# @Author  : WuDiDaBinGe
# @FileName: dataloader.py
# @Software: PyCharm
import collections

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import DataLoader, Dataset
import torch
from config.config import Topic_Config
from transformers import BertTokenizer


def load_data(dir_path, vobsize=30355, test=False):
    dataset = pd.read_csv(dir_path, sep=',')
    # 为'，'和‘！’编号
    dataset['text'] = dataset['text'].map(lambda a: a.replace('，', str(vobsize - 1)))
    dataset['text'] = dataset['text'].map(lambda a: a.replace('！', str(vobsize - 2)))
    dataset['text'] = dataset['text'].map(lambda a: a.split(" "))
    dataset['text'] = dataset['text'].map(lambda a: [int(num) for num in a])
    if test is False:
        # 处理层级标签
        dataset['1-label'] = dataset['label'].map(lambda a: int(a.split('-')[0]))
        dataset['2-label'] = dataset['label'].map(lambda a: int(a.split('-')[1]))
    return dataset


def load_bert_data(dir_path, test=False):
    dataset = pd.read_csv(dir_path, sep=',')
    if test is False:
        # 处理层级标签
        # 想在tf.data.Dataset.map()里添加额外的参数，就要用lambda表达式
        # 也就是dataset = dataset.map(lambda x: decode_example(x, resize_height, resize_width, num_class))
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
        self.dataset = dataset
        self.id_arr = np.asarray(self.dataset.iloc[:, 0])
        self.text_arr = np.asarray(self.dataset.iloc[:, 1])
        self.test = test
        if self.test is False:
            self.first_label_arr = np.asarray(self.dataset.iloc[:, 3])
            self.second_label_arr = np.asarray(self.dataset.iloc[:, 4])
        self.device = device
        if config.embedding_pretrained_model is not None:
            self.vob = config.embedding_pretrained_model.wv.key_to_index.keys()
            # # 加入PAD 字符
            # self.vob.append(0)

    def __getitem__(self, item):
        id_ = self.id_arr[item]
        id_ = torch.tensor(id_).to(self.device)
        token_ids = self.text_arr[item]
        # 处理word embedding预训练的
        if self.config.embedding_pretrained_model is not None:
            token_ids_temp = []
            # 如果不在word embedding中的token 则去掉
            for index, token_id in enumerate(token_ids):
                if token_id in self.vob:
                    # 用0号作为padding embedding多加入了一行， 所以index需要+1
                    token_ids_temp.append(self.config.embedding_pretrained_model.wv.key_to_index[token_id] + 1)
            token_ids = token_ids_temp
        # padding and truncated
        padding_len = self.config.pad_size - len(token_ids)
        if padding_len >= 0:
            token_ids = token_ids + [0] * padding_len
        else:
            token_ids = token_ids[:self.config.pad_size]
        token_ids = torch.tensor(token_ids).to(self.device)
        if self.test is False:
            first_label = self.first_label_arr[item]
            second_label = self.second_label_arr[item]
            first_label = torch.tensor(first_label - 1).to(self.device)
            second_label = torch.tensor(second_label - 1).to(self.device)
            return token_ids, first_label, second_label
        else:
            return id_, token_ids

    def __len__(self):
        return len(self.id_arr)


class Topic_Dataset(Dataset):
    def __init__(self, topic_config, dataset, test=False):
        self.config = topic_config
        self.dataset = dataset
        self.id_arr = np.asarray(self.dataset.iloc[:, 0])
        # 这里的每个元素都是string
        self.text_arr = np.asarray(self.dataset.iloc[:, 1])
        # 将句子转成bert token的形式
        self.tokenizer = BertTokenizer.from_pretrained(self.config.vocab_path)
        if test is False:
            self.first_label_arr = np.asarray(self.dataset.iloc[:, 3])
            self.second_label_arr = np.asarray(self.dataset.iloc[:, 4])
        self.tf_idf = self.get_data_bow()
        self.test = test

    def __getitem__(self, item):
        tf_idf = self.tf_idf[item].to(self.config.device)
        if self.test is False:
            first_label = self.first_label_arr[item]
            second_label = self.second_label_arr[item]
            first_label = torch.tensor(first_label - 1).to(self.config.device)
            second_label = torch.tensor(second_label - 1).to(self.config.device)
            return tf_idf, first_label, second_label
        else:
            return tf_idf

    def __len__(self):
        return len(self.tf_idf)

    def get_data_bow(self):
        # tokenizer 什么参数都不加 不会进行截断 只会加上[cls] 和 [seq]
        row_token_ids = [self.tokenizer(text)['input_ids'][1:-2] for text in self.text_arr]
        vob_size = self.tokenizer.vocab_size
        # 获取文档的词频向量
        bows = [self.tf_score(token_ids, vob_size) for token_ids in row_token_ids]
        # 获取文档的TF-IDF 向量
        # 存放key-> 单词 value:有多少篇文档包含该单词 从而计算tf-idf
        tf_idf_trans = TfidfTransformer()
        tf_idf = tf_idf_trans.fit_transform(bows)
        tf_idf = tf_idf.toarray()
        return torch.from_numpy(tf_idf).float()

    def tf_score(self, data, min_length):
        """
        返回文章的单词词频向量
        :param data: [123,343,1233,2323,~,121]
        :param min_length: 词表的大小
        :return:data的词频向量
        """
        return np.bincount(data, minlength=min_length)


if __name__ == '__main__':
    config = Topic_Config(dataset='../dataset')
    all_set = load_bert_data(config.train_path)
    dataset = Topic_Dataset(topic_config=config, dataset=all_set)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for iter, data in enumerate(dataloader):
        print(data)
        if iter == 3:
            break
