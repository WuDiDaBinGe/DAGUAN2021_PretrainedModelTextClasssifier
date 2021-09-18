# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 上午8:53
# @Author  : WuDiDaBinGe
# @FileName: dataloader.py
# @Software: PyCharm
"""
短文本效果可能较差
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import Topic_Config
from model.wae import WAE
import numpy as np
from dataloader.dataloader import load_bert_data, Topic_Dataset
from utils.topic_quality import evaluate_topic_quality


class W_LDA(object):
    def __init__(self, topic_config_, dataset):
        self.config = topic_config_
        self.bow_dim = dataset.tokenizer.vocab_size
        self.n_topic = topic_config_.n_topic
        self.dropout = topic_config_.dropout
        self.wae = WAE(encode_dims=[self.bow_dim, 1024, 512, self.n_topic],
                       decode_dim=[self.n_topic, 512, self.bow_dim],
                       dropout=topic_config_.dropout,
                       nonlin='relu').to(topic_config_.device)

    def train(self, train_data, test_data):
        bert_tokenizer = train_data.tokenizer
        train_iter = DataLoader(train_data, batch_size=self.config.batch_size, drop_last=True)
        optimizer = torch.optim.Adam(self.wae.parameters(), lr=self.config.lr)
        loss, rec_loss, mmd = 0, 0, 0
        for i in range(self.config.epoch):
            train_tqdm = tqdm(train_iter)
            self.wae.train()
            for iter_num, data in enumerate(train_tqdm):
                optimizer.zero_grad()
                bows, first_label, second_label = data
                bows_recon, theta_q = self.wae(bows)
                theta_prior = self.wae.sample(dist=self.config.dist, batch_size=self.config.batch_size,
                                              ori_data=bows).to(self.config.device)
                logs_softmax = torch.log_softmax(bows_recon, dim=1)
                rec_loss = -1.0 * torch.sum(logs_softmax * bows)
                mmd = self.wae.mmd_loss(theta_q, theta_prior, device=self.config.device, t=0.1)
                s = torch.sum(bows) / len(bows)
                lamb = (5.0 * s * torch.log(torch.tensor(1.0 * bows.shape[-1])) / torch.log(torch.tensor(2.0)))
                mmd = mmd * lamb
                loss = rec_loss + mmd * self.config.beta
                loss.backward()
                optimizer.step()
                train_tqdm.set_description(f'Epoch {i}')
                train_tqdm.set_postfix({'loss': loss.item() / len(bows), "mmd_loss": mmd.item() / len(bows),
                                        'rec_loss': loss.item() / len(bows)})
            if (i + 1) % 10 == 0:
                self.config.writer.add_scalars("Train/Loss",
                                               {"loss": loss / len(bows), "rec_loss": rec_loss / len(bows),
                                                "mmd_loss": mmd / len(bows)}, i)
                topic_coherence, topic_diversity = self.evaluate(bert_tokenizer=bert_tokenizer, test_data=test_data)
                self.config.writer.add_scalars("Test/topic_quality",
                                               {'topic_coherence': topic_coherence, 'topic_diversity': topic_diversity},
                                               i)

            if (i + 1) % 100 == 0:
                checkpoint = {
                    'wae': self.wae.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i,
                    'param': {
                        'bow_dim': self.bow_dim,
                        'n_topic': self.n_topic,
                        'dist': self.config.dist,
                        'dropout': self.config.dropout
                    }
                }
                torch.save(checkpoint, self.config.save_path)
                print(f'Epoch {(i + 1):>3d}\tLoss:{loss/len(bows):<.7f}')
                print('\n'.join([str(lst) for lst in self.show_topic_words(bert_tokenizer)]))
                print('=' * 30)

    def evaluate(self, bert_tokenizer, test_data):
        topic_words = self.show_topic_words(bert_tokenizer)
        test_docs = [doc.split(" ") for doc in test_data.text_arr]
        return evaluate_topic_quality(topic_words, test_docs)

    def inference_by_bow(self, doc_bow):
        if isinstance(doc_bow, np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.to(self.config.device)
        with torch.no_grad():
            self.wae.eval()
            theta = F.softmax(self.wae.encode(doc_bow), dim=1)
            return theta.detach().cpu().numpy()

    def inference(self):
        """
        inference on a dataset
        :return:
        """
        pass

    def show_topic_words(self, bert_tokenizer, topic_id=None, topK=20):
        self.wae.eval()
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.config.device)
        word_dist = self.wae.decode(idxes)
        word_dist = F.softmax(word_dist, dim=1)
        vals, indices = torch.topk(word_dist, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        id2token = bert_tokenizer.ids_to_tokens
        if topic_id is None:
            for i in range(self.n_topic):
                topic_words.append([id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([id2token[idx] for idx in indices[topic_id]])
        return topic_words

    def get_docs_topic(self, train_data):
        self.wae.eval()
        data_loader = DataLoader(train_data, batch_size=512, shuffle=False)
        topics_list = []
        first_label_list = []
        second_label_list = []
        for data_batch in data_loader:
            bows, first_labels, second_labels = data_batch
            embed = self.inference_by_bow(bows)
            topics_list.append(embed)
            first_label_list.append(first_labels.cpu().numpy())
            second_label_list.append(second_labels.cpu().numpy())
        topics_list = np.concatenate(topics_list, axis=0)
        first_label_list = np.concatenate(first_label_list, axis=0)
        second_label_list = np.concatenate(second_label_list, axis=0)
        return topics_list, first_label_list, second_label_list

    def get_topic_word_dist(self, normalize=True):
        self.wae.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.config.device)
            word_dist = self.wae.decode(idxes)  # word_dist: [n_topic, vocab.size]
            if normalize:
                word_dist = F.softmax(word_dist, dim=1)
            return word_dist.detach().cpu().numpy()


if __name__ == '__main__':
    topic_config = Topic_Config(dataset='../dataset')

    train_pd = load_bert_data(topic_config.train_path)
    test_pd = load_bert_data(topic_config.test_path, test=True)

    train_dataset = Topic_Dataset(topic_config, train_pd)
    test_dataset = Topic_Dataset(topic_config, test_pd, test=True)

    w_lda = W_LDA(topic_config_=topic_config, dataset=train_dataset)
    w_lda.train(train_dataset, test_dataset)
