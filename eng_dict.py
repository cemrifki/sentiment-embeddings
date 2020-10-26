#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:45:55 2019

@author: Cem Rıfkı Aydın

A SentiWord entry example:
# POS	ID	PosScore	NegScore	SynsetTerms	Gloss
a	00001740	0.125	0	able#1	(usually followed by `to') having the necessary ...
"""
import os
from collections import Counter

import lexical_interface
import preprocessing

ENG_DICT_PATH = os.path.join("Lexical_Resources", "SentiWordNet.txt")


def get_eng_dict_defs(dict_path):
    """
    This function generates dictionary definitions for English.

    :param dict_path: The path to the SentiWordNet lexicon.
    :type dict_path: str
    :return: English dictionary
    :rtype: dict
    """

    dict_defs = {}
    eng_dict_word_freqs = Counter()
    pre = preprocessing.Preprocessing()
    with open(dict_path, "r") as d:
        for line in d:
            if line[0] == "#" or line[0].isspace():
                continue
            line = line.lower()

            line_spl = line.split('\t')

            synsets = line_spl[4].split()

            synsets = [synset[:-2] for synset in synsets if synset[-1] == "1"]  # The main meaning/synset is captured.

            gloss = line_spl[5]  # The dictionary definition of the entry word.

            gloss_toks = pre.english_tokenize(gloss)

            for gloss_tok in gloss_toks:
                eng_dict_word_freqs[gloss_tok] += 1

            for synset in synsets:
                if synset not in dict_defs:
                    dict_defs[synset] = []
                dict_defs[synset].extend(gloss_toks)
                dict_defs[synset].append(synset)

    dict_defs = lexical_interface.get_dict_except_most_and_least_freq(eng_dict_word_freqs, dict_defs)
    return dict_defs


if __name__ == "__main__":
    dict_defs_ = get_eng_dict_defs(ENG_DICT_PATH)
