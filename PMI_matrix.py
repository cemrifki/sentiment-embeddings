#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın

Most of the code of this module is taken from the Web and is changed a bit to adapt it to this project.
"""

import collections
from collections import Counter
from math import log

import numpy as np

import preprocessing


def get_pmi_dict(texts_tokenized):
    """
    This function builds PMI matrix.

    :param revs: Corpus reviews.
    :type revs: list
    :return: PMI matrix of the corpus.
    :rtype: dict
    """

    cx = Counter()
    cxy = Counter()
    for text in texts_tokenized:
        for x in text:
            cx[x] += 1

    context_dict = preprocessing.Preprocessing().get_all_context_words(texts_tokenized, 5)

    for target_word, cont_words in context_dict.items():

        for word in cont_words:
            cxy[(target_word, word)] += 1
            cxy[(word, target_word)] += 1

        # Alternative: count only 2-grams.
    #     for x, y in zip(text[:-1], text[1:]):
    #         cxy[(x, y)] += 1

    #     # Alternative: count all pairs of words, but don't double count.
    #     for x, y in set(map(tuple, map(sorted, combinations(text, 2)))):
    #         cxy[(x,y)] += 1

    all_words = list(cx.keys())

    # Any bigram containing a unigram that was removed must now be removed.

    for x, y in list(cxy.keys()):
        if x not in cx or y not in cx:
            del cxy[(x, y)]

    # Build unigram <-> index lookup.
    x2i, i2x = {}, {}
    for i, x in enumerate(cx.keys()):
        x2i[x] = i
        i2x[i] = x

    # Sum unigram and bigram counts for computing probabilities.
    # i.e. p(x) = count(x) / sum(all counts).
    sx = sum(cx.values())
    sxy = sum(cxy.values())

    # Accumulate data, rows, and cols to build sparse PMI matrix
    # Recall from the blog post that the PMI value for a bigram with tokens (x, y) is: 
    # PMI(x,y) = log(p(x,y) / p(x) / p(y)) = log(p(x,y) / (p(x) * p(y)))
    # The probabilities are computed on the fly using the sums from above.
    pmi_samples = Counter()
    data, rows, cols = [], [], []
    for (x, y), n in cxy.items():
        rows.append(x2i[x])
        cols.append(x2i[y])
        data.append(max(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)), 0))

        pmi_samples[(x, y)] = data[-1]

    pmi_matrix = collections.OrderedDict({})
    for word1 in all_words:

        list_of_pmis = [0.0 if word1 == word2 else pmi_samples[(word1, word2)] for word2 in all_words]
        max_ = max(list_of_pmis) + 0.01
        list_of_pmis = [x / max_ for x in list_of_pmis]

        word_pmis = [float(pmi) for pmi in list_of_pmis]

        pmi_matrix[word1] = word_pmis

    return pmi_matrix


def prepare_pmi(revs):
    """
    A helper function that builds the PMI matrix based on corpus reviews.

    :param revs: Corpus reviews.
    :type revs: list
    :return: Corpus words and their corresponding vectors.
    :rtype: set, np.arrau
    """
    pmi_dict = get_pmi_dict(revs)

    pmi_dict_keys = pmi_dict.keys()

    pmi_matr = pmi_dict.values()

    pmi_matr = np.array(list(pmi_matr))

    return pmi_dict_keys, pmi_matr
