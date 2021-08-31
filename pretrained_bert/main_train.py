# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午6:55
# @Author  : WuDiDaBinGe
# @FileName: main_train.py
# @Software: PyCharm
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from transformers import BertForMaskedLM
from transformers import AdamW

from dataloader import BertDataset
from pre_config import PreDatasetConfig

# 1) 初始化
torch.distributed.init_process_group(backend="nccl")
# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# 使用BertForMaskedLM来预测一个屏蔽标记
# BertForMaskedLM来预测被mask掉的单词时一定要加特殊字符[ C L S ] 和 [ S E P ] [CLS]和[SEP][CLS]和[SEP]
def train(config, dataset):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 4) 封装之前要把模型移到对应的gpu
    model = BertForMaskedLM(config.bert_config).to(device)
    # 分布式训练需要将bn换成sync_batchnorm进行多卡同步，据说可以进一步加快速度
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    optim = AdamW(model.parameters(), lr=config.lr)
    for epoch in range(config.num_epochs):
        train_dataset.initial()
        # 3）使用DistributedSampler
        train_iter = DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=DistributedSampler(dataset))
        # 显示进度条；保留进度条存在的痕迹，默认为True
        # tqdm()的返回值是一个可迭代对象，迭代的每一个元素就是iterable的每一个参数。该返回值可以修改进度条信息
        loop = tqdm(train_iter, leave=True)
        model.train()
        for batch in loop:
            # 梯度置零，也就是把loss关于weight的导数变成0
            optim.zero_grad()
            input_ids, attention_mask, labels = batch
            # 前向传播求出预测的值
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        checkpoint = {
            'generator': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch
        }
        check_dir = f"./bert_checkpoints"
        if not os.path.exists(check_dir):
            os.mkdir(check_dir)
        # 指定进程号保存
        if local_rank == 0:
            torch.save(checkpoint, os.path.join(check_dir, f"ckpt_{epoch}.pth"))
    model_save_path = './MyBert'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    ## 选择一个进程保存
    if local_rank == 0:
        model.save_pretrained(model_save_path)
        print('Saving model in %s.' % model_save_path)



if __name__ == '__main__':
    # # 初始化
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PreDatasetConfig()
    train_dataset = BertDataset(config, device)
    train(config, train_dataset)
