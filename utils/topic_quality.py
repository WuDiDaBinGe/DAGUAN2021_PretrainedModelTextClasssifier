# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午4:01
# @Author  : WuDiDaBinGe
# @FileName: topic_quality.py
# @Software: PyCharm
import numpy as np


def mimno_topic_coherence(topic_words, docs):
    topic_word_set = set([word for topic in topic_words for word in topic])
    word_doc_dict = {word: set([]) for word in topic_word_set}
    # 遍历文章中的每一个单词
    for doc_id, doc_words in enumerate(docs):
        for word in topic_word_set:
            if word in doc_words:
                word_doc_dict[word].add(doc_id)

    # 计算两个单词在相同文章出中出现的数量
    def co_occur(w1, w2):
        return len(word_doc_dict[w1].intersection(word_doc_dict[w2])) + 1

    scores = []
    for wlst in topic_words:
        s = 0
        # 对每一个主题中每一个词
        for i in range(1, len(wlst)):
            for j in range(0, i):
                if len(word_doc_dict[wlst[j]]) is not 0:
                    s += np.log((co_occur(wlst[i], wlst[j]) + 1.0) / len(word_doc_dict[wlst[j]]))
        scores.append(s)
    return np.mean(scores)


def calc_topic_diversity(topic_words):
    """topic_words is in the form of [[w11,w12,...],[w21,w22,...]] bigger is nicer"""
    # 主题中词的数目
    vocab = set(sum(topic_words, []))
    # 主题中应该有的数目
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div


def evaluate_topic_quality(topic_words, test_data):
    """

    :param topic_words:[[w11,w12,w14],[w21,w22,w23]...[wn1,wn2,wn3]]
    :param test_data:[[w1,w2,w3,],[w1,w2,w3,]]
    :return:
    """
    mimno_tc = mimno_topic_coherence(topic_words=topic_words, docs=test_data)
    topic_diversity = calc_topic_diversity(topic_words)
    return mimno_tc, topic_diversity
