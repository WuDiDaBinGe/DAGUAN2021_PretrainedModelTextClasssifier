# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 上午10:38
# @Author  : WuDiDaBinGe
# @FileName: train_wordEmbedding.py
# @Software: PyCharm
import random
from gensim.models import Word2Vec
import pandas as pd

'''
1. Synonym Replacement (SR): Randomly choose n words from the sentence that are not stop words. Replace each of these words with
one of its synonyms chosen at random.
2. Random Insertion (RI): Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this n times.
3. Random Swap (RS): Randomly choose two words in the sentence and swap their positions. Do this n times.
4. Random Deletion (RD): Randomly remove each word in the sentence with probability p.

不改变超过 1/4 的词汇 alpha = 0.1 maybe best
'''
# 标点
stop_words = [30355 - 1, 30355 - 2]
# word embedding
w2v_model = Word2Vec.load('../process_data/word2vec.model')
w2v_model_keys = w2v_model.wv.key_to_index.keys()


def get_synonyms(word):
    synonyms_list = []
    if word in w2v_model_keys:
        result = w2v_model.wv.most_similar(word, topn=5)
        synonyms_list, confidence = zip(*result)
    return synonyms_list


def synonym_replacement(words, n):
    '''
    随机替换同义词
    :param words:单词列表
    :param n:替换的个数
    :return:增强后的句子
    '''
    new_words = words.copy()
    random_word_list = list(set(new_words) - set(stop_words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


def random_deletion(words, p):
    '''
    随机丢弃
    :param words:单词列表
    :param p:token被保留的概率
    :return:随机丢弃后的句子
    '''
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    # 删除全部单词 随机返回一个
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]
    return new_words


def random_swap(words, n):
    '''
    token 随机交换
    :param words:序列列表
    :param n:n交换次数
    :return:增强后的句子
    '''
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    # sr
    if alpha_sr > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if alpha_ri > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    # rs
    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    # rd
    if p_rd > 0:
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences


def get_label_to_eda(dataset, low_num=100):
    # 统计各类别的样本数
    c = dataset.groupby(['2-label'], as_index=False)['2-label'].agg({'cnt': 'count'})
    c.sort_values('2-label', ascending=True, inplace=True)
    target_labels = c.loc[c['cnt'] < low_num]
    target_labels_list = target_labels.iloc[:, 0].values
    target_labels_num = target_labels.iloc[:, 1].values
    target_label_num_dict = {}
    for index, label in enumerate(target_labels_list):
        target_label_num_dict[label] = int(low_num / target_labels_num[index])
    return target_label_num_dict


if __name__ == '__main__':
    # 数据集增强
    train = pd.read_csv('../dataset/train.csv', sep=',')
    last_id = 14008
    row_train = pd.read_csv('../dataset/train.csv', sep=',')
    # train['text'] = train['text'].map(lambda a: a.split(" "))
    train['1-label'] = train['label'].map(lambda a: int(a.split('-')[0]))
    train['2-label'] = train['label'].map(lambda a: int(a.split('-')[1]))
    # get label need to augment and its single num_aug
    target_label_num_dict = get_label_to_eda(train)

    data_augment_df = {'id': [], 'text': [], 'label': []}
    for index, row in train.iterrows():
        # if row['2-label'] in target_label_num_dict.keys():
        #     row_text = row['text']
        #     target_label_single_aug_num = target_label_num_dict[row['2-label']]
        #     augmented_sentences = eda(row_text, num_aug=target_label_single_aug_num)
        #     if len(augmented_sentences) > 0:
        #         for sent in augmented_sentences:
        #             last_id += 1
        #             data_augment_df['id'].append(last_id)
        #             data_augment_df['text'].append(sent)
        #             data_augment_df['label'].append(row['label'])
        row_text = row['text']
        augmented_sentences = eda(row_text, num_aug=4)
        if len(augmented_sentences) > 0:
            for sent in augmented_sentences:
                last_id += 1
                data_augment_df['id'].append(last_id)
                data_augment_df['text'].append(sent)
                data_augment_df['label'].append(row['label'])
    data_augment_df = pd.DataFrame(data_augment_df)
    total_frame = pd.concat([row_train, data_augment_df])
    # shuffle
    total_frame = total_frame.sample(frac=1.0)
    total_frame.to_csv('../dataset/train_augment.csv', index=False)
    # print(len(eda(words, num_aug=9)))
