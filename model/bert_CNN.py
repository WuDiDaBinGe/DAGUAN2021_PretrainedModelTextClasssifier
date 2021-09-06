# -*- coding: utf-8 -*-
# @Time    : 2021/8/30 下午10:41
# @Author  : WuDiDaBinGe
# @FileName: bert_CNN.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_local)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.second_num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, token_ids, mask):
        outs = self.bert(token_ids, attention_mask=mask)
        sequence_out = outs[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(sequence_out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


