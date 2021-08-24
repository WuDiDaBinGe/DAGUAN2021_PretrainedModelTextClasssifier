# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午1:34
# @Author  : WuDiDaBinGe
# @FileName: process_vob.py
# @Software: PyCharm
import tqdm
input_dir = '/dataset/daguan_data_class/datagrand_2021_unlabeled_data.json'
vocab = {}
not_improve = 0
max_length = -1
with open(input_dir, 'r', encoding='utf-8') as f:
    for doc in f:
        doc_dict = eval(doc)
        doc_vob = doc_dict['title'].split(" ") + doc_dict['content'].split(" ")
        for vob in doc_vob:
            vocab[vob] = 1
        if len(vocab) > max_length:
            max_length = len(vocab)
            not_improve = 0
        else:
            not_improve += 1
        if not_improve > 100000:
            break
vocab = sorted(vocab.keys(), key=lambda d: (len(d), d))
f_out = open('../dataset/vocab.txt', 'w')
f_out.write("\n".join(vocab))
f_out.close()
