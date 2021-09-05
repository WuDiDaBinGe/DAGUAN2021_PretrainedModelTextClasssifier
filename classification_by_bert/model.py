import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel

from Multi_Heads import MultiHeadSelfAttention
from config import Config
from dataloader import load_data, MyDataset


# class Classifier(nn.Module):
#     def __init__(self, config):
#         super(Classifier, self).__init__()
#         self.config = config
#         self.bert = BertModel.from_pretrained(self.config.bert_local)
#         self.ws1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)  # 256->256
#         self.ws2 = nn.Linear(self.config.hidden_size * 2, 1)  # 256->1
#         self.dropout = nn.Dropout(self.config.dropout)
#         self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.second_num_classes)
#         self.softmax = nn.Softmax(dim=1)
#         self.lstm = nn.LSTM(self.config.embedding_dim, self.config.hidden_size, self.config.num_layers,
#                             bidirectional=True,
#                             batch_first=True,
#                             dropout=config.dropout)
#
#     def forward(self, token_ids, mask, ):
#         outputs = self.bert(token_ids, attention_mask=mask)
#         sequence_output = outputs[0]
#         H, _ = self.lstm(sequence_output)
#         self_attention = torch.tanh(self.ws1(self.dropout(H)))
#         self_attention = self.ws2(self.dropout(self_attention)).squeeze()
#         self_attention = self_attention + -10000 * (mask == 0).float()
#         self_attention = self.softmax(self_attention)
#         # 加入attention
#         sent_encoding = torch.sum(H * self_attention.unsqueeze(-1), dim=1)
#         pred = self.classifier(sent_encoding)
#         # pred = self.softmax(sent_encoding)
#         return pred


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_local)
        self.ws1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)  # 1024->1024
        self.ws2 = nn.Linear(self.config.hidden_size * 2, 1)  # 1024->1
        self.dropout = nn.Dropout(self.config.dropout)
        self.pre_classifier = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)  # 1024->512
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.second_num_classes)  # 512->35
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.hidden_size, self.config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)

    def forward(self, token_ids, mask, ):
        outputs = self.bert(token_ids, attention_mask=mask)
        sequence_output = outputs[0]
        H, _ = self.lstm(sequence_output)
        self_attention = torch.tanh(self.ws1(self.dropout(H)))
        self_attention = self.ws2(self.dropout(self_attention)).squeeze()
        self_attention = self_attention + -10000 * (mask == 0).float()
        self_attention = self.softmax(self_attention)
        # 加入attention
        sent_encoding = torch.sum(H * self_attention.unsqueeze(-1), dim=1)
        pre_pred = torch.tanh(self.pre_classifier(self.dropout(sent_encoding)))
        pred = self.classifier(self.dropout(pre_pred))
        # pred = self.softmax(sent_encoding)
        return pred


class ClassifierCNN(nn.Module):
    def __init__(self, config):
        super(ClassifierCNN, self).__init__()
        self.config = config
        self.config.filter_sizes = [1, 3, 5, 7]
        self.config.num_filters = 64
        self.bert = BertModel.from_pretrained(self.config.bert_local)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.config.embedding_dim, self.config.num_filters, k) for k in
             self.config.filter_sizes]
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc_cnn = nn.Linear(self.config.num_filters * len(self.config.filter_sizes), self.config.second_num_classes)
        dim_in = self.config.embedding_dim
        heads_num = 8
        dim_k = dim_v = self.config.embedding_dim
        self.attentions = nn.ModuleList([MultiHeadSelfAttention(dim_in, dim_k, dim_v, heads_num) for _ in
                                         range(len(self.config.filter_sizes))])
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def attention_and_conv_pool(self, x, conv, attention):
        x = attention(x)
        x = x.transpose(1, 2)
        x = F.relu(conv(x))
        x = self.pooling(x).squeeze(2)
        return x

    def forward(self, token_ids, mask):
        outs = self.bert(token_ids, attention_mask=mask)
        sequence_out = outs[0]
        out = torch.cat(
            [self.attention_and_conv_pool(sequence_out, conv, attention) for conv, attention in
             zip(self.convs, self.attentions)], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class ClassifierCNNInPaper(nn.Module):
    def __init__(self, config):
        super(ClassifierCNNInPaper, self).__init__()
        self.config = config
        # 定义卷积核size
        self.config.filter_sizes = [1, 3, 5, 7]
        # 定义卷积核个数
        self.config.num_filters = 64
        # 加载预训练bert
        self.bert = BertModel.from_pretrained(self.config.bert_local)
        # required gradient
        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.config.embedding_dim, self.config.num_filters, k) for k in
             self.config.filter_sizes]
        )
        self.dropout = nn.Dropout(0.1)
        self.fc_cnn = nn.Linear(self.config.num_filters * len(self.config.filter_sizes), self.config.second_num_classes)
        dim_in = self.config.embedding_dim
        heads_num = 8
        dim_k = dim_v = self.config.embedding_dim
        self.attentions = nn.ModuleList([MultiHeadSelfAttention(dim_in, dim_k, dim_v, heads_num) for _ in
                                         range(len(self.config.filter_sizes))])
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def attention_and_conv_pool(self, x, conv, attention):
        x = attention(x)
        x = x.transpose(1, 2)
        x = F.relu(conv(x))
        x = self.pooling(x).squeeze(2)
        x = self.dropout(x)
        return x

    def forward(self, token_ids, mask):
        outs = self.bert(token_ids, attention_mask=mask)
        sequence_out = outs[0]
        sequence_out = self.dropout(sequence_out)
        out = torch.cat(
            [self.attention_and_conv_pool(sequence_out, conv, attention) for conv, attention in
             zip(self.convs, self.attentions)], 1)
        out = self.fc_cnn(out)
        return out


if __name__ == '__main__':
    config = Config(dataset='../dataset')
    train_set = load_data(config.train_path)
    # dev_set = load_data(config.dev_path)
    train_dataset = MyDataset(config=config, dataset=train_set, device=config.device)
    # dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    model = ClassifierCNN(config).to(config.device)
    for inputs in train_dataloader:
        token_ids, masks, first_label, second_label = inputs
        model.zero_grad()
        pred = model(token_ids, masks)
