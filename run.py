# -*- coding: utf-8 -*-
# @Time    : 2021/8/20 下午3:02
# @Author  : WuDiDaBinGe
# @FileName: run.py
# @Software: PyCharm
import torch
from torch.utils.data import DataLoader

from config.config import Config
from dataloader.dataloader import load_data, spilt_dataset_pd, MyDataset
from train.trainer import train
from model.TextRNN_Atten import TextRNN_Att

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    config = Config(dataset='./dataset')
    all_set = load_data(config.train_path)
    train_pd, dev_pd = spilt_dataset_pd(all_set)
    train_dataset = MyDataset(config=config, dataset=train_pd, device=device)
    dev_dataset = MyDataset(config=config, dataset=dev_pd, device=device)
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_iter = DataLoader(dev_dataset, batch_size=1, shuffle=True)
    model = TextRNN_Att(config=config)
    train(config=config, model=model, train_dataset=train_iter, dev_iter=dev_iter)
