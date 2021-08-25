# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午6:55
# @Author  : WuDiDaBinGe
# @FileName: main_train.py
# @Software: PyCharm
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BertForMaskedLM
from transformers import AdamW

from dataloader import BertDataset
from pre_config import PreDatasetConfig

# 使用BertForMaskedLM来预测一个屏蔽标记
# BertForMaskedLM来预测被mask掉的单词时一定要加特殊字符[ C L S ] 和 [ S E P ] [CLS]和[SEP][CLS]和[SEP]
def train(config, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForMaskedLM(config.bert_config).to(device)
    optim = AdamW(model.parameters(), lr=config.lr)
    for epoch in range(config.num_epochs):
        train_dataset.initial()
        train_iter = DataLoader(dataset=dataset, batch_size=config.batch_size)
        # 显示进度条；保留进度条存在的痕迹，默认为True
        # tqdm()的返回值是一个可迭代对象，迭代的每一个元素就是iterable的每一个参数。该返回值可以修改进度条信息
        loop = tqdm(train_iter, leave=True)
        model.train()
        for batch in loop:
            # 梯度置零，也就是把loss关于weight的导数变成0
            optim.zero_grad()
            input_ids, attention_mask, labels = batch
            # 前向传播求出预测的值
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        checkpoint = {
            'generator': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch
        }
        check_dir = f"./bert_checkpoints"
        if not os.path.exists(check_dir):
            os.mkdir(check_dir)
        torch.save(checkpoint, os.path.join(check_dir, f"ckpt_{epoch}.pth"))
    model_save_path = './MyBert'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    model.save_pretrained('./MyBert')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PreDatasetConfig()
    train_dataset = BertDataset(config, device)
    train(config, train_dataset)
