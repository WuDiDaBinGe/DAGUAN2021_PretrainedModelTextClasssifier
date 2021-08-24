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

from pretrained_bert.dataloader import BertDataset
from pretrained_bert.pre_config import PreDatasetConfig


def train(config, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForMaskedLM(config.bert_config).to(device)
    optim = AdamW(model.parameters(), lr=config.lr)
    for epoch in range(config.num_epochs):
        train_dataset.initial()
        train_iter = DataLoader(dataset=dataset, batch_size=config.batch_size)
        loop = tqdm(train_iter, leave=True)
        model.train()
        for batch in loop:
            optim.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
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
