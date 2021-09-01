import time

import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataloader import load_data, spilt_dataset_pd, MyDataset
from model import Classifier
from config import Config

loss_weight = []


# gamma=0，效果等同交叉熵函数
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.ones(class_num, 1) * alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# 二级标签35个
loss_func = FocalLoss(class_num=35)


# class MultiCEFocalLoss(torch.nn.Module):
#     def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
#         super(MultiCEFocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         pt = F.softmax(predict, dim=1) # softmmax获取预测概率
#         class_mask = F.one_hot(target, 5) #获取target的one hot编码
#         ids = target.view(-1, 1)
#         alpha = self.alpha[ids.data.view(-1)]
#         # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
#         probs = (pt * class_mask).sum(1).view(-1, 1)
#         # 利用onehot作为mask，提取对应的pt
#         log_p = probs.log()
#         # 同样，原始ce上增加一个动态权重衰减因子
#         loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
#
# loss_func = MultiCEFocalLoss(class_num=35,alpha=0.25)

def train(config, model, train_dataset, dev_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
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
            loss = loss_func(pred, second_label)
            # 加flood方法，试图优化过拟合
            # flood = (loss - 0.35).abs() + 0.35
            total_loss += loss.item()
            loss.backward()
            # flood_loss += flood
            optimizer.step()
            data.set_description(f'Epoch {epoch}')
            data.set_postfix(loss=loss.item())
        (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
        writer.add_scalar("loss/train", total_loss, epoch)
        # writer.add_scalar("loss/flood", flood_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss, epoch)
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


def evaluate(config, model, dev_dataset):
    y_true, y_pred = [], []
    model.eval()
    total_loss = 0
    for data in tqdm(dev_dataset):
        token_ids, masks, first_label, second_label = data
        model.zero_grad()
        pred = model(token_ids, masks)
        total_loss += F.cross_entropy(pred, second_label).item()
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


if __name__ == '__main__':
    config = Config(dataset='/home/wsj/dataset/2021达观杯')
    loss_weight = [0] * config.second_num_classes
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    all_set = load_data(config.train_path)
    train_set, dev_set = spilt_dataset_pd(all_set)
    train_dataset = MyDataset(config=config, dataset=train_set, device=config.device)
    dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    model = Classifier(config).to(config.device)
    train(config, model, train_dataloader, dev_dataloader)
