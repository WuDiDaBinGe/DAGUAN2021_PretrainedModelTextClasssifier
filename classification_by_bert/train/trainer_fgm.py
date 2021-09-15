# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 下午3:07
# @Author  : WuDiDaBinGe
# @FileName: trainer_fgm.py
# @Software: PyCharm
import time

import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from classification_by_bert.model.FGM import FGM
from model.bert_CNN import BertCNN, Bert4LayerCNN, Bert4Layer
from classification_by_bert.dataloader import load_data, MyDataset
from classification_by_bert.model.model import Classifier
from classification_by_bert.config import Config
from classification_by_bert.train.trainer import evaluate
from classification_by_bert.model.focalloss import FocalLoss

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def train(config, model, train_dataset, dev_dataset, loss_function):
    # (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
    # return
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
    for epoch in range(config.num_epochs):
        total_loss = 0
        # flood_loss = 0
        model.train()
        data = tqdm(train_dataset, leave=True)
        fgm = FGM(model)
        for inputs in data:
            token_ids, masks, first_label, second_label = inputs
            pred = model(token_ids, masks)
            # loss = F.cross_entropy(pred, second_label)
            loss = loss_function(pred, second_label)
            total_loss += loss.item()
            # 正常回传 正常的grad
            loss.backward()
            # 对抗训练
            fgm.attack(emb_name='word_embeddings')
            pred_adv = model(token_ids, masks)
            loss_adv = loss_function(pred_adv, second_label)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore(emb_name='word_embeddings')
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
            data.set_description(f'Epoch {epoch}')
            data.set_postfix(loss=loss.item())
        (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset,
                                                                        loss_function=loss_function)
        writer.add_scalar("loss/train", total_loss / len(train_dataset), epoch)
        # writer.add_scalar("loss/flood", flood_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss / len(dev_dataset), epoch)
        writer.add_scalars("performance/f1", {'macro_f1': macro_f1, 'micro_f1': micro_f1}, epoch)
        writer.add_scalar("performance/precision", precision, epoch)
        writer.add_scalar("performance/recall", recall, epoch)
        if prev_best_perf < macro_f1:
            prev_best_perf = macro_f1
            improve_epoch = epoch
            torch.save(model.state_dict(), config.save_path)
            print("model saved!!!")
        elif epoch - improve_epoch >= config.require_improvement:
            print("model didn't improve for a long time! So break!!!")
            break
    writer.close()


if __name__ == '__main__':
    config = Config(dataset='../../dataset/', name='Bert4layerCNN-fgm-aeda')

    all_set = load_data(config.train_argument_path)
    train_dataset = MyDataset(config=config, dataset=all_set, device=config.device)

    dev_set = load_data(config.dev_path)
    dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    # 重要性采样
    sample_weights = 1.0 / torch.tensor(config.every_class_nums, dtype=torch.float)
    train_targets = train_dataset.get_all_classes()
    sample_weights = sample_weights[train_targets]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # shuffle 是 false
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, sampler=sampler)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    model = Bert4LayerCNN(config).to(config.device)
    train(config, model, train_dataloader, dev_dataloader, loss_function=F.cross_entropy)
