import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataloader import load_data, MyDataset
from model.bert_CNN import Bert4LayerCNN


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
    print("Classification Report \n", classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    return macro_scores, total_loss, micro_scores[2]


if __name__ == '__main__':
    config = Config(dataset='../dataset', name='Bert4LayerCNN')
    dev_set = load_data(config.dev_path)
    dev_dataset = MyDataset(config=config, dataset=dev_set, device=config.device)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    model = Bert4LayerCNN(config).to(config.device)
    model.load_state_dict(torch.load(
        "../dataset/saved_dict/Bert4layerCNN09-08_15.42.ckpt"))
    evaluate(config, model, dev_dataloader)
