#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import logging
import multiprocessing
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import constants


class Word2vec:
    """
    This class is used to model the word2vec vectors of corpus words.
    """
    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels

    def get_vecs(self):
        """
        This helper method returns word2vec embeddings of words.

        :return: A dict whose keys are words and values are their corresponding word2vec vectors.
        :rtype:
        """
        return self.get_word2vecs()

    def get_word2vecs(self):
        """
        This helper method generates word2vec embeddings from a corpus.

        :return: Corpus words and their word2vec embeddings.
        :rtype: dict
        """
        model = Word2Vec(self.revs, size=constants.EMBEDDING_SIZE, window=constants.CONTEXT_WINDOW_SIZE,
                         min_count=1, workers=multiprocessing.cpu_count())
        vecs = model.wv.vectors.tolist()

        words = list(model.wv.vocab)

        res_dict = dict(zip(words, vecs))

        return res_dict


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print(
            "Please provide two arguments, first one is path to the revised corpus,"
            " second one is path to the output file for model.")
        print("Example command: python3 word2vec.py wiki.tr.txt trmodel")
        sys.exit()

    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    model_ = Word2Vec(LineSentence(inputFile), size=200, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model_.wv.save_word2vec_format(outputFile, binary=False)
