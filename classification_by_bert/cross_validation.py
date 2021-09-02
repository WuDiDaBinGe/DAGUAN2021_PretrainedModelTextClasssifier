from classification_by_bert.config import Config
from classification_by_bert.dataloader import load_data, spilt_dataset_pd

config = Config(dataset='../dataset')
all_set = load_data(config.train_path)
# 添加loss weight
# calculate_loss_weight()
train_set, dev_set = spilt_dataset_pd(all_set)
