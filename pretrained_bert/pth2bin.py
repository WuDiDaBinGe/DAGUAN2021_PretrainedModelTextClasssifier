import torch
from transformers import BertForMaskedLM

from pre_config import PreDatasetConfig

config = PreDatasetConfig()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = BertForMaskedLM(config.bert_config)
data = torch.load('./gpu_bert_checkpoints/ckpt_0.pth', map_location='cuda:0')
# model.load_state_dict(['generator'])
print()
