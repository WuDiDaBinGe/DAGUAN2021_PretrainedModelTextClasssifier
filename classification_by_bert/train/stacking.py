# -*- coding: utf-8 -*-
# @Time    : 2021/9/5 下午2:48
# @Author  : WuDiDaBinGe
# @FileName: stacking.py
# @Software: PyCharm
import os
import time
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from classification_by_bert.dataloader import load_data, MyDataset
from classification_by_bert.model.model import Classifier, ClassifierCNN
from model.second_liner_classifier import LinearClassifier
from classification_by_bert.config import Config
from classification_by_bert.model.focalloss import FocalLoss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler  # 混合精度降低显存使用


# loss_func = ASLSingleLabel()

def train(config, model, train_dataset, dev_dataset, loss_function):
    # (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
    # return
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    save_path = config.save_path.split('.ckpt')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = save_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()) + '.ckpt'
    # 实例化GraScalaer()对象
    scaler = GradScaler()
    for epoch in range(config.num_epochs):
        total_loss = 0
        # flood_loss = 0
        model.train()
        data = tqdm(train_dataset, leave=True)
        for inputs in data:
            token_ids, masks, first_label, second_label = inputs
            model.zero_grad()
            # 前向过程中(model + loss)开启 autocast
            with autocast():
                pred = model(token_ids, masks)
                # loss = F.cross_entropy(pred, second_label)
                loss = loss_function(pred, second_label)
            # 加flood方法，试图优化过拟合
            # flood = (loss - 0.35).abs() + 0.35
            total_loss += loss.item()
            # loss.backward()
            # # flood_loss += flood
            # optimizer.step()
            # Scales loss 为了梯度方法
            scaler.scale(loss).backward()
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)
            # 准备着，看是否要增大scaler
            scaler.update()
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
            torch.save(model.state_dict(), file_path)
            print("model saved!!!")
        elif epoch - improve_epoch >= config.require_improvement:
            print("model didn't improve for a long time! So break!!!")
            break
    writer.close()


def evaluate(config, model, dev_dataset, loss_function=None):
    y_true, y_pred = [], []
    model.eval()
    total_loss = 0
    # 加速和节省gpu空间
    with torch.no_grad():
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
    with torch.no_grad():
        for data in tqdm(test_iter):
            if len(data) == 3:
                id, token_ids, masks = data
            else:
                token_ids, masks, first_label, second_label = data
            model.zero_grad()
            pred = model(token_ids, masks).cpu()
            pred_all.append(pred)
    res = pred_all[0]
    for index in range(1, len(pred_all)):
        res = torch.cat((res, pred_all[index]), dim=0)
    return res


def get_oof(config, clf_name, train_data, dev_data, test_data, loss_function):
    '''
    获取一个模型clf的训练集、测试集表示 以方便二级分类器训练
    :param config:设置
    :param clf_name:分类模型名称
    :param train_data:训练集
    :param dev_data:验证集
    :param test_data:测试集数据
    :return:该模型在训练集的表示，该模型在测试集的表示
    '''

    kf = KFold(n_splits=config.kfold)
    oof_train = torch.zeros((len(train_data), config.second_num_classes))

    oof_test = torch.zeros((len(test_data), config.second_num_classes))
    oof_test_skf = torch.zeros((5, len(test_data), config.second_num_classes))

    oof_dev = torch.zeros((len(dev_data), config.second_num_classes))
    oof_dev_skf = torch.zeros((5, len(dev_data), config.second_num_classes))
    # 每一折
    dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)
    for i, (train_index, valid_index) in enumerate(kf.split(train_data)):
        # 每一折模型都要重新训练
        clf_model = eval(clf_name).to(config.device)
        print(train_index, valid_index)
        train_data_sub = Subset(train_data, train_index)
        valid_data_sub = Subset(train_data, valid_index)
        train_dataloader = DataLoader(train_data_sub, batch_size=config.batch_size)

        valid_dataloader = DataLoader(valid_data_sub, batch_size=config.batch_size)
        # 训练
        # train(config, clf_model, train_dataloader, dev_dataloader, loss_function=loss_function)
        # 预测验证集
        train_rep = predict(clf_model, valid_dataloader)
        oof_train[valid_index] = train_rep
        oof_test_skf[i, :] = predict(clf_model, test_dataloader)
        oof_dev_skf[i, :] = predict(clf_model, dev_dataloader)
        del clf_model
        torch.cuda.empty_cache()
    oof_test[:] = torch.mean(oof_test_skf, dim=0)
    oof_dev[:] = torch.mean(oof_dev_skf, dim=0)
    return oof_train, oof_test, oof_dev


def covert_tensor_to_file(data, file_path):
    result = np.array(data)
    result = result.reshape(-1, 35)
    np.savetxt(file_path, result)


def train_linear_classifier(config, linear_model, train_tensor, dev_tensor):
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    prev_best_perf = 0
    improve_epoch = 0
    train_ids = TensorDataset(train_tensor)
    train_iter = DataLoader(dataset=train_ids, batch_size=config.batch_size)

    dev_ids = TensorDataset(dev_tensor)
    dev_iter = DataLoader(dataset=dev_ids, batch_size=config.batch_size)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=config.learning_rate)
    for epoch in range(100):
        total_loss = 0
        data = tqdm(train_iter, leave=True)
        for inputs in data:
            x_data, y_label = data
            linear_model.zero_grad()
            pred = linear_model(x_data)
            loss = F.cross_entropy(pred, y_label)
            total_loss += loss.item()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            data.set_description(f'Epoch {epoch}')
            data.set_postfix(loss=loss.item())
        (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, linear_model, dev_iter)
        writer.add_scalar("loss/train", total_loss / len(train_ids), epoch)
        # writer.add_scalar("loss/flood", flood_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss / len(dev_ids), epoch)
        writer.add_scalars("performance/f1", {'macro_f1': macro_f1, 'micro_f1': micro_f1}, epoch)
        writer.add_scalar("performance/precision", precision, epoch)
        writer.add_scalar("performance/recall", recall, epoch)
        if prev_best_perf < macro_f1:
            prev_best_perf = macro_f1
            improve_epoch = epoch
            torch.save(linear_model.state_dict(), config.save_path)
            print("model saved!!!")
        elif epoch - improve_epoch >= config.require_improvement:
            print("model didn't improve for a long time! So break!!!")
            break
    writer.close()

def evaluate_linear(linear_model, dev_iter):
    y_true, y_pred = [], []
    linear_model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dev_iter):
            x_data, y_label = data
            linear_model.zero_grad()
            pred = linear_model(x_data)
            total_loss += F.cross_entropy(pred, y_label).item()
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
    print("Classification Report \n", classification_report(y_true, y_pred, digits=4))
    # print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    return macro_scores, total_loss, micro_scores[2]

def inference_by_linear(config, linear_model, test_tensor):
    result = {'id': [], 'label': []}
    linear_model.eval()
    for data in tqdm(test_tensor):
        id, token_ids, masks = data
        result['id'].append(id.item())
        pred = linear_model(token_ids, masks)
        # pred = pred.squeeze()
        _, predict = torch.max(pred, 1)
        if torch.cuda.is_available():
            predict = predict.cpu()
        second_predic = predict
        res = str(config.label_rel[second_predic.item() + 1]) + '-' + str(second_predic.item() + 1)
        result['label'].append(res)
    df = pd.DataFrame(result)
    df.to_csv(config.submit_path, index=False)


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
    train_proba = torch.zeros((len(train_dataset), len(model_name_list), config.second_num_classes))
    # 验证集结果
    dev_proba = torch.zeros((len(dev_dataset), len(model_name_list), config.second_num_classes))
    # 测试集结果
    test_proba = np.zeros((len(test_dataset), len(model_name_list), config.second_num_classes))
    for index in range(len(model_list)):
        print("Start train {} model".format(model_name_list[index]))
        train_p, test_p, dev_p = get_oof(config=config, clf_name=model_list[index], train_data=train_dataset,
                                         dev_data=dev_dataset,
                                         test_data=test_dataset,
                                         loss_function=FocalLoss(class_num=config.second_num_classes, alpha=2))

        train_proba[:, index, :] = train_p
        test_proba[:, index, :] = test_p
        dev_proba[:, index, :] = dev_p
        print("End train {} model".format(model_name_list[index]))
    covert_tensor_to_file(train_proba, f'stacking_train.txt')
    covert_tensor_to_file(test_proba, f'stacking_test.txt')


if __name__ == '__main__':
    config = Config(dataset='../../dataset', name='class')
    config.batch_size = 24
    config.num_epochs = 1
    model_list = ['Classifier(config)', 'ClassifierCNN(config)']
    # loss_function_list = []
    stacking_model_train(config, model_list)
