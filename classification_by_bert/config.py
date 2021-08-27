import torch


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = "classification_by_bert"
        self.train_path = dataset + '/datagrand_2021_train.csv'  # 训练集
        self.dev_path = dataset + '/datagrand_2021_train.csv'  # 验证集
        self.test_path = dataset + '/datagrand_2021_test.csv'  # 测试集
        self.submit_path = dataset + '/submit.csv'

        self.vocab_path = dataset + '/vocab.txt'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 50  # 若超过50epoch效果还没提升，则提前结束训练
        self.second_num_classes = 35  # level 2 label
        self.num_epochs = 1000  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 350  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率

        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.label_rel = [0, 1, 2, 2, 1, 3, 2, 4, 6, 1, 1, 2, 5, 6, 2, 6, 7, 2, 8, 6, 6, 6, 5, 9, 5, 2, 10, 8, 6, 6, 5,
                          6, 6, 2, 6, 5]

        self.bert_local = dataset + '/MyBert'  # 预训练模型
        self.embedding_dim = 768
