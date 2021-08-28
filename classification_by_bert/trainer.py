import time

import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataloader import load_data, spilt_dataset_pd, MyDataset
from model import Classifier
from config import Config


def train(config, model, train_dataset, dev_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    prev_best_perf = 0
    improve_epoch = 0
    for epoch in range(config.num_epochs):
        total_loss = 0
        model.train()
        data = tqdm(train_dataset, leave=True)
        for inputs in data:
            token_ids, masks, first_label, second_label = inputs
            model.zero_grad()
            pred = model(token_ids, masks)
            loss = F.cross_entropy(pred, second_label)
            # 加flood方法，试图优化过拟合
            flood = (loss-0.35).abs()+0.35
            total_loss += flood.item()
            flood.backward()
            optimizer.step()
            data.set_description(f'Epoch {epoch}')
            data.set_postfix(loss=flood.item())
        (precision, recall, macro_f1, _), dev_loss, micro_f1 = evaluate(config, model, dev_dataset)
        writer.add_scalar("loss/train", total_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss, epoch)
        writer.add_scalars("performance/f1", {'macro_f1': macro_f1, 'micro_f1': micro_f1}, epoch)
        writer.add_scalar("performance/precision", precision, epoch)
        writer.add_scalar("performance/recall", recall, epoch)
        if prev_best_perf < macro_f1:
            prev_best_perf = macro_f1
            improve_epoch = epoch
            torch.save(model.state_dict(), config.save_path)
            print("model saved!!!")
        elif epoch - improve_epoch > config.require_improvement:
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
    print("MACRO: ", macro_scores)
    print("MICRO: ", micro_scores)
    print("Classification Report \n", classification_report(y_true, y_pred))
    # 将信息输出到tensorboard中
    print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    return macro_scores, total_loss, micro_scores[2]


if __name__ == '__main__':
    config = Config(dataset='/home/wsj/dataset/2021达观杯')
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    all_set = load_data(config.train_path)
    train_set, dev_set = spilt_dataset_pd(all_set)
    train_dataset = MyDataset(config=config, dataset=train_set, device=config.device)
    dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True)
    model = Classifier(config).to(config.device)
    train(config, model, train_dataloader, dev_dataloader)
