#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın

This module is the interface designed for different embedding types.

"""

from clustering import Clustering
from supervised_features import SupervisedFeatures
from svd_U import CorpusSvdU
from svd_U import LexicalSvdU
from vec_operations import VecOperations
from word2vec import Word2vec

embedding_methods = {"corpus_svd": CorpusSvdU,
                     "lexical_svd": LexicalSvdU,
                     "supervised": SupervisedFeatures,
                     "ensemble": VecOperations,
                     "clustering": Clustering,
                     "word2vec": Word2vec}
