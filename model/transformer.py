# -*- coding: utf-8 -*-
# @Time    : 2021/8/27 下午12:39
# @Author  : WuDiDaBinGe
# @FileName: transformer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000 ** (i // 2 * 2 / embed)) for i in range(embed)] for pos in range(pad_size)])
        # [:,0::2] start:0 step:2
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        # Why dropout??
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)  # [batch_size,seq, dim_V]
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        assert dim_model % num_head == 0
        self.num_head = num_head
        self.dim_k = dim_model // num_head
        # 使用全连接表示矩阵 多个矩阵 （这里其实用一个大矩阵 表示了多个小矩阵）
        self.fc_Q = nn.Linear(dim_model, dim_model)
        self.fc_K = nn.Linear(dim_model, dim_model)
        self.fc_V = nn.Linear(dim_model, dim_model)

        self.attention = Scaled_Dot_Product_Attention()

        self.fc = nn.Linear(num_head * self.dim_k, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.normal = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        # Q\K\V: [batch*num_head,seq,dim_k]
        Q = Q.view(batch_size * self.num_head, -1, self.dim_k)
        K = K.view(batch_size * self.num_head, -1, self.dim_k)
        V = V.view(batch_size * self.num_head, -1, self.dim_k)
        context = self.attention(Q, K, V)  # [batch_size*head_num, seq, dim_k]
        context = context.view(batch_size, -1, self.dim_k * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        # 残差链接的位置
        out = out + x
        out = self.normal(out)
        return out
