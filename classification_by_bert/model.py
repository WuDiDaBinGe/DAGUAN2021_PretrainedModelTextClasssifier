import torch
from torch import nn
from transformers import BertModel


# bert+LSTM+attention
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


# 老刘用这个
# 模型结构同上，lstm隐藏层维度由128→512
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