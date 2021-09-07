# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 下午10:50
# @Author  : WuDiDaBinGe
# @FileName: second_liner_classifier.py
# @Software: PyCharm
import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, num_class, model_num):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(num_class * model_num, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        '''
        :param x: 每个模型拼接后的概率分布
        :return:
        '''
        out = self.fc(x)
        out = self.dropout(out)
        return out
