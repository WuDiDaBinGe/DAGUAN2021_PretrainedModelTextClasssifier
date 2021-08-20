# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午3:20
# @Author  : WuDiDaBinGe
# @FileName: TextRNN_Atten.py
# @Software: PyCharm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class TextRNN_Att(nn.Module):
    def __init__(self, config):
        super(TextRNN_Att, self).__init__()
        if config.embedding_pretrained:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True,
                            dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        # head one
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.fc_first = nn.Linear(config.hidden_size, config.first_num_classes)
        # head two
        self.fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.fc_second = nn.Linear(config.hidden_size, config.second_num_classes)

    def forword(self, x):
        # x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        # first label
        first_label = self.fc1(out)
        first_label = self.fc_first(first_label)  # [128, 64]
        # two label
        second_label = self.fc2(out)
        second_label = self.fc_second(second_label)
        return first_label, second_label