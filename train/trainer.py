# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 下午11:06
# @Author  : WuDiDaBinGe
# @FileName: trainer.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from tensorboardX import SummaryWriter

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_dataset, dev_iter, test_iter=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()
        for i, data in enumerate(train_dataset):
            token_ids, first_labels, second_labels = data
            first_outputs, second_outputs = model(token_ids)
            model.zero_grad()
            first_loss = F.cross_entropy(first_outputs, first_labels)
            second_loss = F.cross_entropy(second_outputs, second_labels)
            # total_loss = first_loss + second_loss
            total_loss = second_loss
            second_loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 验证集测试
                first_true = first_labels.data.cpu()
                first_predic = torch.max(first_outputs.data, 1)[1].cpu()
                train_first_f1 = metrics.f1_score(first_true, first_predic, average='micro')

                second_true = second_labels.data.cpu()
                second_predic = torch.max(second_outputs.data, 1)[1].cpu()
                train_second_f2 = metrics.f1_score(second_true, second_predic, average='micro')

                # train_f1 = metrics.f1_score(100 * first_true + second_true, 100 * first_predic + second_predic,
                #                             average='micro')
                train_f1 = train_second_f2
                dev_f1, dev_loss = evaluate(config, model, dev_iter)
                model.train()
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = '*'
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train f1: {2:>6.2%},  Val Loss: {3:>5.2},  Val f1: {4:>6.2%}'
                print(msg.format(total_batch, total_loss.item(), train_f1, dev_loss, dev_f1, improve))
                writer.add_scalar("loss/train", total_loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("level_loss/first_label", first_loss.item(), total_batch)
                writer.add_scalar("level_loss/second_label", second_loss.item(), total_batch)
                writer.add_scalar("f1/train", train_f1, total_batch)
                writer.add_scalar("f1/dev", dev_f1, total_batch)
                writer.add_scalar("level_f1/level_f1", train_first_f1, total_batch)
                writer.add_scalar("level_f1/level_f2", train_second_f2, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 早停
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()


def model_test(config, model, test_iter):
    pass


def evaluate(config, model, dev_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for token_ids, first_labels, second_labels in dev_iter:
            first_outputs, second_outputs = model(token_ids)
            first_loss = F.cross_entropy(first_outputs, first_labels)
            second_loss = F.cross_entropy(second_outputs, second_labels)
            total_loss = first_loss + second_loss
            # loss_total += total_loss
            loss_total += second_loss

            first_predic = torch.max(first_outputs.data, 1)[1].cpu()
            second_predic = torch.max(second_outputs.data, 1)[1].cpu()
            # labels_all = np.append(labels_all, (first_labels * 100 + second_labels).cpu())
            # predict_all = np.append(predict_all, (first_predic * 100 + second_predic))
            labels_all = np.append(labels_all, second_labels.cpu())
            predict_all = np.append(predict_all, second_predic)
    f1_score = metrics.f1_score(labels_all, predict_all, average='micro')
    return f1_score, loss_total.cpu() / len(dev_iter)
