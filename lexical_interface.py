#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın

This module is the lexical interface that helps generate lexicon entries and their meanings.
This is implemented for both Turkish and English.
It provides additional functions, such as noise removal for dictionaries.

"""
import copy
import os

import numpy as np

import constants
import eng_dict
import supervised_features
import turk_dict

HIERARCHY_HEIGHT = 2  # If this is initialised as 3, the below variable
# "HIERARCHY_COEFFS" should have 3 values in a decreasing order.

HIERARCHY_COEFFS = [1.0, 0.5]

dict_lang = {"english": eng_dict.get_eng_dict_defs,
             "turkish": turk_dict.get_turkish_dict_defs}

ENG_DICT_PATH = os.path.join("Lexical_Resources", "SentiWordNet.txt")

TURK_DICT_PATH = os.path.join("Lexical_Resources", 'Turkish_Dictionary_Disamb')

LANG_DICT_PATH = {"english": ENG_DICT_PATH,
                  "turkish": TURK_DICT_PATH}


def get_dict_except_most_and_least_freq(frequency_dict_, dict_defs):
    """
    This function returns the dictionary definitions stripped off noise words for English and Turkish.

    :param frequency_dict_: Dict that contains the frequency information of words in the lexicon.
    :type frequency_dict_: dict
    :param dict_defs: All dictionary definitions.
    :type dict_defs: dict
    :return: The dictionary definitions of entry words.
    :rtype: dict
    """
    most_and_least_freq_words = get_most_and_least_freq_words(frequency_dict_)

    res_dict = {}

    for word, dict_def in dict_defs.items():
        res_dict[word] = []
        normal_dict_words = [dict_def_word for dict_def_word in dict_def
                             if dict_def_word not in most_and_least_freq_words]
        res_dict[word].extend(normal_dict_words)

    return res_dict


def get_most_and_least_freq_words(frequency_dict_):
    """
    This function returns the most and least frequent words that can be considered noise.

    :param frequency_dict_: Dictionary that holds the frequency information of words.
    :type frequency_dict_: dict
    :return: Words that are frequent in these lexicons.
    :rtype: set
    """
    size = len(frequency_dict_)
    frequency_el = int(size / 100)

    most_freq = frequency_dict_.most_common(frequency_el)
    least_freq = frequency_dict_.most_common()[:-frequency_el - 1:-1]

    union_set = set([k[0] for k in most_freq]).union(set(k[0] for k in least_freq))

    return union_set


def create_lev_hierarchy(dict_defs, no_of_levs):
    """
    This function generates a lexical hierarchy. For example, the entry word is atop the hierarchy.
    The words occurring in its dictionary definition are at the second highest level, and so on.

    :param dict_defs: Dictionary definitions.
    :type dict_defs: dict
    :param no_of_levs: The number of levels used in generating the hierarchy tree.
    :type no_of_levs: int
    :return: Hierarchy tree.
    :rtype: dict
    """
    hier_dict_defs = {}
    for k, v in dict_defs.items():
        hier_dict_defs[k] = {}
        hier_dict_defs[k][0] = copy.deepcopy(v)
        hier_dict_defs[k][0].append(k)
        tmp_v = copy.deepcopy(v)
        # tmp_words = []
        for i in range(1, no_of_levs):
            tmp_lev_words = []
            for word in tmp_v:
                if word in dict_defs:
                    tmp_lev_words.extend(copy.deepcopy(dict_defs[word]))

            hier_dict_defs[k][i] = tmp_lev_words
            tmp_v = copy.deepcopy(tmp_lev_words)
    return hier_dict_defs


def create_dict_hierarchy(path, no_of_levs):
    """
    A helper function that is used in generating hierarchy trees.

    :param path: The path to the lexicon, which is the Turkish or English lexicon in this case.
    :type path: str
    :param no_of_levs: The number of levels (height) of the hierarchy tree.
    :type no_of_levs: int
    :return: Hierarchy tree for the dictionary definitions.
    :rtype: dict
    """

    get_dict_defs = dict_lang[constants.LANG]

    dict_defs = get_dict_defs(path)  # get_Turkish_dict_defs(path)

    hier_dict = create_lev_hierarchy(dict_defs, no_of_levs)

    return hier_dict


def create_dict_matr(corp_revs, path, no_of_levs):
    """
    The function that generates matrix for dictionary definitions with respect to the corpus reviews.

    :param corp_revs: Corpus reviews.
    :type corp_revs: list
    :param path: Path to dictionaru.
    :type path: str
    :param no_of_levs: The height of the hierarchy tree.
    :type no_of_levs: int
    :return: Dictionary matrix with respect to corpus words that is of the dict type.
    :rtype: dict
    """
    corp_dict_matr = {}

    corp_words = set().union(*corp_revs)

    corp_dict = get_corp_dict_hierarchy(corp_words, path, no_of_levs)

    all_words = get_all_dict_words(corp_dict)

    for k, v in corp_dict.items():
        row = []

        for word in all_words:

            found_word = False
            for i in range(0, no_of_levs):
                lev_meaning_words = v[i]
                if word in lev_meaning_words:
                    found_word = True
                    break

            val = HIERARCHY_COEFFS[i] if found_word else 0.0
            row.append(val)

        corp_dict_matr[k] = row

    return corp_dict_matr


def get_all_dict_words(dict_):
    """
    This function gets all the word entries of the dictionary.

    :param dict_: Dictionary words and their definitions.
    :type dict_: dict
    :return: All the word entries.
    :rtype: list
    """
    words = set([])

    for k, v in dict_.items():
        for lev, meaning_words in v.items():
            words = words.union(meaning_words)

    return list(words)


def get_corp_dict_hierarchy(corp_words, path, no_of_levs):
    """
    This function obtains all the dictionary definition hierarchies for corpus words.

    :param corp_words: Corpus words.
    :type corp_words: set and list both accepted.
    :param path: Path to the dictionary.
    :type path: str
    :param no_of_levs: The height of the hierarchy tree.
    :type no_of_levs: int
    :return: Hierarchy trees built by dictionary definition with respect to corpus words.
    :rtype: dict
    """
    corp_dict_hier = {}
    dict_hier = create_dict_hierarchy(path, no_of_levs)
    for word in corp_words:
        if word in dict_hier:
            corp_dict_hier[word] = dict_hier[word]

    return corp_dict_hier


def get_lexical_superv_dict(dict_, delta_idfs):
    """
    This function combines the lexical and supervised components.

    :param dict_: Dictionary definition vectors for corpus words.
    :type dict_: dict
    :param delta_idfs: Delta-idf scores of words obtained from the training corpus.
    :type delta_idfs: dict
    :return: Vectors obtained by multiplying dictionary definition vectors by the delta-idf score of the word entry.
    :rtype: dict
    """

    res_dict = {}
    for word, vals in dict_.items():
        if word in delta_idfs:
            vals = np.array(vals)
            vals *= delta_idfs[word]
            vals = vals.tolist()
        else:
            vals = np.random.uniform(-1.0, 1.0, len(vals)).tolist()
        res_dict[word] = vals
    return res_dict


def get_lexical_dict(corp_revs, *labels):
    """
    This functions generates lexical vectors (be it supervised or not) with respect to corpus words.

    :param corp_revs: Corpus reviews.
    :type corp_revs: list
    :param labels: Labels (positive or negative) present in the training dataset.
    :type labels: list
    :return: Vectors of the corpus words that contain lexical and supervised information.
    :rtype: dict
    """

    dict_matr = create_dict_matr(corp_revs, LANG_DICT_PATH[constants.LANG], HIERARCHY_HEIGHT)

    sf = supervised_features.SupervisedFeatures(corp_revs, *labels)
    delta_idfs = sf.delta_idf_scores

    superv_dict_matr = get_lexical_superv_dict(dict_matr, delta_idfs)

    return superv_dict_matr
