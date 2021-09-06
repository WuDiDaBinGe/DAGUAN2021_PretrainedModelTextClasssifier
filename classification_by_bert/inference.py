import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import load_data, MyDataset
from classification_by_bert.model.model import Classifier, ClassifierCNN
from config import Config


def inference(config, model, test_iter):
    result = {'id': [], 'label': []}
    model.eval()
    for data in tqdm(test_iter):
        id, token_ids, masks = data
        result['id'].append(id.item())
        pred = model(token_ids, masks)
        # pred = pred.squeeze()
        _, predict = torch.max(pred, 1)
        if torch.cuda.is_available():
            predict = predict.cpu()
        second_predic = predict
        res = str(config.label_rel[second_predic.item() + 1]) + '-' + str(second_predic.item() + 1)
        result['label'].append(res)
    df = pd.DataFrame(result)
    df.to_csv(config.submit_path, index=False)


def model_voting_inference(config, model_list, test_iter):
    result = {'id': [], 'label': []}
    for model in model_list:
        model.eval()

    for data in tqdm(test_iter):
        id, token_ids, masks = data
        result['id'].append(id.item())
        pred_list = []
        for model in model_list:
            pred_list.append(model(token_ids, masks))
        pred_total = pred_list[0]
        for i in range(1, len(pred_list)):
            pred_total += pred_list[i]
        pred_total = pred_total / len(model_list)
        # pred = pred.squeeze()
        _, predict = torch.max(pred_total, 1)
        if torch.cuda.is_available():
            predict = predict.cpu()
        second_predic = predict
        res = str(config.label_rel[second_predic.item() + 1]) + '-' + str(second_predic.item() + 1)
        result['label'].append(res)
    df = pd.DataFrame(result)
    df.to_csv(config.submit_path, index=False)


if __name__ == '__main__':
    config = Config(dataset='../dataset', name="classifier")
    test = load_data(dir_path=config.test_path, test=True)
    test_dataset = MyDataset(config=config, dataset=test, device=config.device, test=True)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ClassifierCNN(config).to(config.device)
    model.load_state_dict(torch.load(r"../dataset/saved_dict/classification_by_bert09-04_10.04.ckpt"))

    model_asl = Classifier(config).to(config.device)
    model_asl.load_state_dict(
        torch.load(r"../dataset/saved_dict/0.57_ACL-loss_baseline/saved_dict/classification_by_bert.ckpt"))

    model_focal = Classifier(config).to(config.device)
    model_focal.load_state_dict(
        torch.load(r'../dataset/saved_dict/0.568_focal_loss_baseline/saved_dict/classification_by_bert.ckpt'))
    model_list = [model_asl, model_focal, model]
    model_voting_inference(config, model_list, test_iter)
