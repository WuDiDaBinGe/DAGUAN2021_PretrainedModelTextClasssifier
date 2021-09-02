# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 上午10:38
# @Author  : WuDiDaBinGe
# @FileName: train_wordEmbedding.py
# @Software: PyCharm
import random
from gensim.models import Word2Vec

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


def random_delete(words, p):
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


if __name__ == '__main__':
    words = '4685 12369 8139 24457 1989 104 13839 16328 233 4198 17568 4920 8223 4898 17281 3328 1317 30354 25483 19121 14547 19269 11414 22443 20448 5697 20089 26856 19121 12235 13462 14950 30354 25174 12846 27363 11497 2281 25203 7469 23182 20726 24058 8773 24073 8147 22410 15584 4873 27908 7659 17281 28438 9192 19121 15469'
    words = words.split(" ")
    words = [int(word) for word in words]
    n = int(len(words) * 0.1)
    print(synonym_replacement(words, n))
    print(random_swap(words, n))
    print(random_delete(words, 0.1))
    print(random_insertion(words, n))
