#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import os
from collections import Counter

import lexical_interface
import preprocessing

TURK_DICT_PATH = os.path.join("Lexical_Resources", 'Turkish_Dictionary_Disamb')

WORD_SEP = "wordsep"
MEANING_MARK = "meaning"

turk_dict_word_freqs = Counter()


def remove_noise_char(word):
    """
    Parser and disambiguator can express the information about which meaning a word has.
    This helper function removes this information.

    :param word: A corpus word.
    :type word: str
    :return: The corpus word stripped off the information mentioned above.
    :rtype: str
    """

    return word[:word.index("(i")] if "(i" in word else word


def get_turkish_dict_defs(dict_path):
    """
    This function extracts the dictionary definitions of words in Turkish.

    :param dict_path: The path to the Turkish dictionary that is in parsed and disambiguated form.
    :type dict_path: str
    :return: A dict containing words and their dictionary definitions.
    :rtype: dict
    """
    
    pre = preprocessing.Preprocessing()

    all_mwes = pre.get_turkish_idioms()
    
    dict_definitions = {}

    for filename in os.listdir(dict_path):
        file = os.path.join(dict_path, filename)
        with open(file, "r") as f:
            meaning = False
            dict_entry_word = ""
            for line in f:
                line = line.lower()
                if line[0] == "[" or line[0] == "]":
                    continue
                if WORD_SEP in line:
                    # The below if block performs some preprocessing operations.
                    if dict_entry_word in dict_definitions:
                        # The below line of code could have been optimised.
                        dict_def = " ".join(list(dict_definitions[dict_entry_word]))
                        upd_def = pre.replace_mwe(dict_def, all_mwes)
                        upd_def = pre.capture_and_update_consec_negs(upd_def)
                        dict_definitions[dict_entry_word] = set(upd_def)
                    dict_entry_word = ""
                    meaning = False
                    continue

                if MEANING_MARK in line:
                    meaning = True
                    if dict_entry_word[-1] == '_':
                        dict_entry_word = dict_entry_word[:-1]
                    dict_entry_word = remove_noise_char(dict_entry_word)
                    if dict_entry_word not in dict_definitions:
                        dict_definitions[dict_entry_word] = set([])
                        # dict_definitions[dict_entry_word].add(dict_entry_word)
                    continue

                if not meaning and "[" in line:
                    if "lH[Adj+With]" in line.split()[1]:
                        dict_entry_word += line.split()[0]
                    else:
                        dict_entry_word += line.split("[")[0].split()[1]

                elif meaning and "[" in line:
                    meaning_word = line.split("[")[0].split()[1]
                    meaning_word = remove_noise_char(meaning_word)
                    turk_dict_word_freqs[meaning_word] += 1
                    dict_definitions[dict_entry_word].add(meaning_word)

    dict_definitions = lexical_interface.get_dict_except_most_and_least_freq(turk_dict_word_freqs, dict_definitions)
    return dict_definitions


if __name__ == "__main__":
    print(get_turkish_dict_defs(TURK_DICT_PATH))
