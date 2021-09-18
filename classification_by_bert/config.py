import time

import torch


class Config(object):
    """配置参数"""

    def __init__(self, dataset, name):
        self.model_name = name
        self.train_argument_path = dataset + '/train_augment.csv'  # 训练集
        self.train_path = dataset + '/train.csv'  # 训练集
        self.dev_path = dataset + '/dev.csv'  # 验证集
        self.test_path = dataset + '/datagrand_2021_test.csv'  # 测试集
        self.submit_path = dataset + '/submit.csv'

        self.vocab_path = dataset + '/vocab.txt'  # 词表
        self.roberta_vocab_path = dataset + '/roberta_pretrained/roberta_pretrained_vocab'

        self.bert_local = dataset + '/MyBert'  # 预训练模型
        self.roberta_local = dataset + '/roberta_6'

        self.save_path = dataset + '/saved_dict/' + self.model_name + time.strftime('%m-%d_%H.%M',
                                                                                    time.localtime()) + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.1  # 随机失活
        self.require_improvement = 50  # 若超过100epoch效果还没提升，则提前结束训练
        self.second_num_classes = 35  # level 2 label
        self.num_epochs = 150  # epoch数
        self.batch_size = 16  # mini-batch大小
        if name == "BertCNN":
            self.pad_size = 350  # 每句话处理成的长度(短填长切)，最长344
        else:
            self.pad_size = 350  # 每句话处理成的长度(短填长切)，最长344
        self.learning_rate = 1e-5  # 学习率

        self.hidden_size = 512  # lstm中隐藏层的维度
        self.num_layers = 2  # lstm层数
        self.label_rel = [0, 1, 2, 2, 1, 3, 2, 4, 6, 1, 1, 2, 5, 6, 2, 6, 7, 2, 8, 6, 6, 6, 5, 9, 5, 2, 10, 8, 6, 6, 5,
                          6, 6, 2, 6, 5]

        self.embedding_dim = 768
        self.filter_sizes = (2, 3, 4, 5)
        self.num_filters = 256
        # 数据集增强的
        # self.every_class_nums = [870, 595, 795, 640, 196, 662, 512, 237, 1523, 516, 283, 160, 145, 222, 513, 184, 141,
        #                          187, 142, 184, 212, 204, 196, 208, 190, 196, 210, 173, 257, 1019, 130, 174, 464, 250,
        #                          297]
        # train dataset
        self.every_class_nums = [870, 595, 795, 640, 14, 662, 512, 237, 1523, 516, 283, 160, 145, 222, 513, 23, 141, 17, 142, 23, 106, 204, 14,
        52, 190, 14, 42, 173, 257, 1019, 130, 87, 464, 125, 297]

        self.kfold = 5
