# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 下午11:31
# @Author  : WuDiDaBinGe
# @FileName: test.py
# @Software: PyCharm
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from classification_by_bert.bert_CNN import BertCNN
from classification_by_bert.config import Config
from classification_by_bert.dataloader import load_data, MyDataset, spilt_dataset_pd
from classification_by_bert.trainer import evaluate
from classification_by_bert.model import Classifier


def model_voting(model_list, dev_iter):
    y_true, y_pred = [], []
    for model in model_list:
        model.eval()
    total_loss = 0
    for data in tqdm(dev_iter):
        token_ids, masks, first_label, second_label = data
        pred_list = []
        for model in model_list:
            model.zero_grad()
            pred_list.append(model(token_ids, masks))
        pred_total = pred_list[0]
        for i in range(1, len(pred_list)):
            pred_total += pred_list[i]
        total_loss += F.cross_entropy(pred_total/len(pred_list), second_label).item()
        pred = pred_total.squeeze()
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


if __name__ == '__main__':
    config = Config(dataset='../dataset', name="classifier")

    all_set = load_data(config.dev_path)
    dev_dataset = MyDataset(config=config, dataset=all_set, device=config.device)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)

    model = BertCNN(config).to(config.device)
    model.load_state_dict(torch.load(r"../dataset/saved_dict/bert_cnn09-01_09.09_best.ckpt"))

    model_asl = Classifier(config).to(config.device)
    model_asl.load_state_dict(
        torch.load(r"../dataset/saved_dict/0.57_ACL-loss_baseline/saved_dict/classification_by_bert.ckpt"))

    model_focal = Classifier(config).to(config.device)
    model_focal.load_state_dict(
        torch.load(r'../dataset/saved_dict/0.568_focal_loss_baseline/saved_dict/classification_by_bert.ckpt'))
    model_list = [model_focal, model_asl, model]

    model_voting(model_list, dev_dataloader)
