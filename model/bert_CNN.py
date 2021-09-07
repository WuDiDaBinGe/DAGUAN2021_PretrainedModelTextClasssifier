# -*- coding: utf-8 -*-
# @Time    : 2021/8/30 下午10:41
# @Author  : WuDiDaBinGe
# @FileName: bert_CNN.py
# @Software: PyCharm
import torch
from torch.utils.data import DataLoader

from classification_by_bert.config import Config
from torch import nn
from transformers import BertModel, AutoConfig
import torch.nn.functional as F

from classification_by_bert.dataloader import load_data, MyDataset


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
        out = self.bert(token_ids, attention_mask=mask)
        sequence_out = out[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(sequence_out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class Bert4LayerCNN(nn.Module):
    def __init__(self, config):
        super(Bert4LayerCNN, self).__init__()
        self.config = config
        config_ = AutoConfig.from_pretrained(config.bert_local)
        # 获取每层的输出
        config_.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained(self.config.bert_local, config=config_)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 4 * config.embedding_dim)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.second_num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, token_ids, mask):
        out = self.bert(token_ids, attention_mask=mask)
        out = torch.stack(out[2])
        out = torch.cat(
            (out[-1], out[-2], out[-3], out[-4]), dim=-1
        )
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class Bert4Layer(nn.Module):
    def __init__(self, config):
        super(Bert4Layer, self).__init__()
        self.config = config
        bert_config = AutoConfig.from_pretrained(config.bert_local)
        bert_config.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained(self.config.bert_local, config=bert_config)
        self.linear = nn.Linear(4 * config.embedding_dim, config.second_num_classes)

    def forward(self, token_ids, mask):
        outputs = self.bert(token_ids, mask)
        outputs = torch.stack(outputs[2])
        concatenate_pooling = torch.cat(
            (outputs[-1], outputs[-2], outputs[-3], outputs[-4]), dim=-1
        )
        concatenate_pooling = concatenate_pooling[:, 0]
        output = self.linear(concatenate_pooling)
        return output


if __name__ == '__main__':
    config = Config(dataset='../dataset', name="Bert4Layers")
    train_set = load_data(config.train_path)
    # dev_set = load_data(config.dev_path)
    train_dataset = MyDataset(config=config, dataset=train_set, device=config.device)
    # dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    model = Bert4LayerCNN(config).to(config.device)
    for inputs in train_dataloader:
        token_ids, masks, first_label, second_label = inputs
        model.zero_grad()
        pred = model(token_ids, masks)
