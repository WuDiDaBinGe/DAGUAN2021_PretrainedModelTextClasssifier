import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import load_data, MyDataset
from model import Classifier
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


if __name__ == '__main__':
    config = Config(dataset='../dataset')
    test = load_data(dir_path=config.test_path, test=True)
    test_dataset = MyDataset(config=config, dataset=test, device=config.device, test=True)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = Classifier(config).to(config.device)
    model.load_state_dict(torch.load(r"../dataset/saved_dict/classification_by_bert08-28_17.21_flood.ckpt"))
    inference(config, model, test_iter)
