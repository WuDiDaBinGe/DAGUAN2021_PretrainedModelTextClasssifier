# -*- coding: utf-8 -*-
# @Time    : 2021/8/20 下午6:41
# @Author  : WuDiDaBinGe
# @FileName: inference.py
# @Software: PyCharm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config.config import Config
from dataloader.dataloader import load_data, MyDataset
from model.TextRNN_Atten import TextRNN_Att


def load_model(model_path):
    model = TextRNN_Att(config=config).to(device=device)
    model.load_state_dict(torch.load(model_path))
    return model


def inference(config, model, test_iter):
    result = {'id': [], 'label': []}
    model.eval()
    for iter, data in enumerate(test_iter):
        id, token_ids = data
        result['id'].append(id.item())
        first_outputs, second_outputs = model(token_ids)
        first_predic = torch.max(first_outputs.data, 1)[1].cpu()
        second_predic = torch.max(second_outputs.data, 1)[1].cpu()
        res = str(config.label_rel[second_predic.item()+1]) + '-' + str(second_predic.item() + 1)
        result['label'].append(res)
    df = pd.DataFrame(result)
    df.to_csv(config.submit_path, index=False)


if __name__ == '__main__':
    config = Config(dataset='/home/wsj/dataset/2021达观杯', embedding='./dataset/pretrained_wordEmbedding/word2vec.model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = load_data(dir_path=config.test_path, test=True)
    test_dataset = MyDataset(config=config, dataset=test, device=device, test=True)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = load_model(config.save_path)
    inference(config, model, test_iter)
