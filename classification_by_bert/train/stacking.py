# -*- coding: utf-8 -*-
# @Time    : 2021/9/5 下午2:48
# @Author  : WuDiDaBinGe
# @FileName: stacking.py
# @Software: PyCharm
import time
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from classification_by_bert.dataloader import load_data, MyDataset
from classification_by_bert.model.model import Classifier
from classification_by_bert.config import Config
from classification_by_bert.model.focalloss import FocalLoss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


# loss_func = ASLSingleLabel()

def train(config, model, train_dataset, dev_dataset, loss_function):
    # (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
    # return
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    save_path = config.save_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())
    for epoch in range(config.num_epochs):
        total_loss = 0
        # flood_loss = 0
        model.train()
        data = tqdm(train_dataset, leave=True)
        for inputs in data:
            token_ids, masks, first_label, second_label = inputs
            model.zero_grad()
            pred = model(token_ids, masks)
            # loss = F.cross_entropy(pred, second_label)
            loss = loss_function(pred, second_label)
            # 加flood方法，试图优化过拟合
            # flood = (loss - 0.35).abs() + 0.35
            total_loss += loss.item()
            loss.backward()
            # flood_loss += flood
            optimizer.step()
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
            torch.save(model.state_dict(), save_path)
            print("model saved!!!")
        elif epoch - improve_epoch >= config.require_improvement:
            print("model didn't improve for a long time! So break!!!")
            break
    writer.close()


def evaluate(config, model, dev_dataset, loss_function=None):
    y_true, y_pred = [], []
    model.eval()
    total_loss = 0
    for data in tqdm(dev_dataset):
        token_ids, masks, first_label, second_label = data
        model.zero_grad()
        pred = model(token_ids, masks)
        if loss_function is None:
            total_loss += F.cross_entropy(pred, second_label).item()
        else:
            total_loss += loss_function(pred, second_label).item()
        pred = pred.squeeze()
        _, predict = torch.max(pred, 1)
        if torch.cuda.is_available():
            predict = predict.cpu()
            second_label = second_label.cpu()
        y_pred += list(predict.numpy())
        temp_true = list(second_label.numpy())
        y_true += temp_true

    macro_scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_scores = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # print("MACRO: ", macro_scores)
    # print("MICRO: ", micro_scores)
    print("Classification Report \n", classification_report(y_true, y_pred))
    # print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    return macro_scores, total_loss, micro_scores[2]


def predict(model, test_iter):
    '''
    模型预测函数
    :param model: 训练的分类器
    :param test_iter:测试集数据
    :return: 返回模型预测的测试集
    '''
    model.eval()
    pred_all = []
    for data in tqdm(test_iter):
        token_ids, masks, first_label, second_label = data
        model.zero_grad()
        pred = model(token_ids, masks)
        pred_all.append(pred)
    res = pred_all[0]
    for index in range(1, len(pred_all)):
        res = torch.cat((res, pred_all[index]), dim=0)
    return res


def get_oof(config, clf, train_data, dev_data, test_data, loss_function):
    '''
    获取一个模型clf的训练集、测试集表示 以方便二级分类器训练
    :param config:设置
    :param clf:分类模型
    :param train_data:训练集
    :param dev_data:验证集
    :param test_data:测试集数据
    :return:该模型在训练集的表示，该模型在测试集的表示
    '''
    kf = KFold(n_splits=config.kfold, random_state=2001)
    oof_train = torch.zeros((len(train_data), config.second_num_classes))
    oof_test = torch.zeros((len(test_data), config.second_num_classes))
    oof_test_skf = torch.zeros((5, len(test_data), config.second_num_classes))
    # 每一折
    dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)
    for i, (train_index, valid_index) in enumerate(kf.split(train_data)):
        train_data = Subset(train_data, train_index)
        valid_data = Subset(train_data, valid_index)
        train_dataloader = DataLoader(train_data, batch_size=config.batch_size)

        valid_dataloader = DataLoader(valid_data, batch_size=config.batch_size)
        # 训练
        train(config, clf, train_dataloader, dev_dataloader, loss_function=loss_function)
        # 预测验证集
        oof_train[valid_index] = predict(clf, valid_dataloader)
        oof_test_skf[i, :] = predict(clf, test_dataloader)

    oof_test[:] = torch.mean(oof_test_skf, dim=0)
    return oof_train, oof_test


def stacking_model_train(config, model_list):
    model_name_list = [model.split('(')[0] for model in model_list]
    # 加载训练集 验证 以及测试集
    train_set = load_data(config.train_path)
    train_dataset = MyDataset(config=config, dataset=train_set, device=config.device)
    dev_set = load_data(config.dev_path)
    dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    test = load_data(dir_path=config.test_path, test=True)
    test_dataset = MyDataset(config=config, dataset=test, device=config.device, test=True)
    # 保存以及分类器的结果
    # 保存训练集结果
    train_proba = np.zeros((len(train_dataset), len(model_name_list)))
    train_proba = pd.DataFrame(train_proba)
    train_proba.columns = model_name_list
    # 测试集结果
    test_proba = np.zeros((len(test_dataset), len(model_name_list)))
    test_proba = pd.DataFrame(test_proba)
    test_proba.columns = model_name_list
    for index in range(len(model_list)):
        print("Start train {} model".format(model_name_list[index]))
        model = eval(model_list[index]).to(config.device)
        train_p, test_p = get_oof(config=config, clf=model, train_data=train_dataset, dev_data=dev_dataset,
                                  test_data=test_dataset,
                                  loss_function=FocalLoss(class_num=config.second_num_classes, alpha=2))



if __name__ == '__main__':
    config = Config(dataset='/home/wsj/dataset/2021达观杯')
    loss_weight = [0] * config.second_num_classes

    all_set = load_data(config.train_path)
    # train_set, dev_set = spilt_dataset_pd(all_set)
    all_dataset = MyDataset(config=config, dataset=all_set, device=config.device)
    # all_dataloader = DataLoader(all_dataset, batch_size=config.batch_size, shuffle=True)
    model = Classifier(config).to(config.device)
    # model.load_state_dict(torch.load(r"/home/wsj/dataset/2021达观杯/augment_focal_loss/saved_dict/classification_by_bert.ckpt"))
    train(config, model, all_dataset)
