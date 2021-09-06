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

loss_weight = []

# 二级标签35个
loss_func = FocalLoss(class_num=35)


# loss_func = ASLSingleLabel()

# K折交叉验证
def train(config, model, all_dataset):
    # (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
    # return
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
    flag = True
    # 固定五折数据
    # 当shuffle为True时，random_state会影响索引的顺序，从而控制每折交叉的随机性
    # 相当于random_state用来控制随机状态，表示是否固定随机起点，也就是随机种子
    kf = KFold(n_splits=config.kfold, random_state=2001, shuffle=True)
    epoch = 0

    for train_index, valid_index in kf.split(all_dataset):
        # 每一折训练30个epoch
        for i in range(30):
            total_loss = 0
            train_data = Subset(all_dataset, train_index)
            valid_data = Subset(all_dataset, valid_index)
            train_dataloder = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            valid_dataloder = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)

            model.train()
            data = tqdm(train_dataloder, leave=True)
            for inputs in data:
                token_ids, masks, first_label, second_label = inputs
                model.zero_grad()
                pred = model(token_ids, masks)
                # loss = F.cross_entropy(pred, second_label)
                temp_loss = loss_func(pred, second_label)
                total_loss += temp_loss.item()
                temp_loss.backward()
                optimizer.step()
                data.set_description(f'Epoch {epoch}')
                data.set_postfix(loss=temp_loss.item())

            (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, valid_dataloder)
            writer.add_scalar("loss/train", total_loss, epoch)
            writer.add_scalar("loss/dev", dev_loss, epoch)
            writer.add_scalars("performance/f1", {'macro_f1': macro_f1, 'micro_f1': micro_f1}, epoch)
            writer.add_scalar("performance/precision", precision, epoch)
            writer.add_scalar("performance/recall", recall, epoch)
            epoch = epoch + 1
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
    # train_set, dev_set = spilt_dataset_pd(all_set)
    all_dataset = MyDataset(config=config, dataset=all_set, device=config.device)
    # all_dataloader = DataLoader(all_dataset, batch_size=config.batch_size, shuffle=True)
    model = Classifier(config).to(config.device)
    # model.load_state_dict(torch.load(r"/home/wsj/dataset/2021达观杯/augment_focal_loss/saved_dict/classification_by_bert.ckpt"))
    train(config, model, all_dataset)
