# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 下午1:34
# @Author  : WuDiDaBinGe
# @FileName: process_vob.py
# @Software: PyCharm
import h5py

unlabel_path = '/dataset/daguan_data_class/datagrand_2021_unlabeled_data.json'


def get_vob(input_dir):
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


def get_total_rows_valid_row(input_dir):
    total_rows = 0
    valid_rows = 0

    with open(input_dir, 'r', encoding='utf-8') as f:
        for doc in f:
            doc_dict = eval(doc)
            total_rows += 1
            doc_vob = doc_dict['title'].split(" ") + doc_dict['content'].split(" ")
            if len(doc_vob) > 5:
                valid_rows += 1
    return total_rows, valid_rows


def covert_to_hdf5(input_dir, out_dir):
    valid_rows = 0
    h5_f = h5py.File(out_dir, 'w')
    dataset = h5_f.create_dataset('Mytxt', shape=(10,))
    with open(input_dir, 'r', encoding='utf-8') as f:
        for doc in f:
            doc_dict = eval(doc)
            title = doc_dict['title']
            context = doc_dict['content']
            full_text = title + " " + context
            doc_vob = doc_dict['title'].split(" ") + doc_dict['content'].split(" ")
            if len(doc_vob) > 5:
                # dataset[valid_rows] = full_text
                h5_f.attrs[str(valid_rows)] = full_text
                valid_rows += 1

    h5_f.close()


if __name__ == '__main__':
    covert_to_hdf5('/dataset/daguan_data_class/newdata.json', '/home/yxb/newdata.h5')
    g = h5py.File('test.h5', 'r')
    print(g.attrs['0'])
