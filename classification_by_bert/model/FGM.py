# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 下午2:54
# @Author  : WuDiDaBinGe
# @FileName: FGM.py
# @Software: PyCharm
import torch


class FGM():
    def __init__(self, model):
        super(FGM, self).__init__()
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
