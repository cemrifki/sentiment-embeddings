#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın

Constant parameters to be leveraged across the program.

"""

import os

COMMAND = "cross_validate"

# Dimension size of embeddings
EMBEDDING_SIZE = 100

#Language can be either "turkish" or "english"
LANG = "turkish"

DATASET_PATH = os.path.join("input", "Sentiment_dataset_turk.csv")  # Sentiment_dataset_tr.csv; test_tr.csv

CONTEXT_WINDOW_SIZE = 5

""" Word embedding type can be one of the following:

corpus_svd: SVD - U vector
lexical_svd: Dictionary, SVD - U vector
supervised: Four context delta idf score vector
ensemble: Combination of the above three embeddings
clustering: Clustering vector
word2vec: word2vec

"""
EMBEDDING_TYPE = "ensemble"


# Number of cross-validation
CV_NUMBER = 10

# Concatenation of the maximum, average, and minimum delta-idf polarity scores with the average document vector.
USE_3_REV_POL_SCORES = True

# The below variables are used only if the command "train_and_test_separately" is chosen.
TRAINING_FILE_PATH = os.path.join("input", "Turkish_twitter_train.csv")
TEST_FILE_PATH = os.path.join("input", "Turkish_twitter_test.csv")

# The below file name for the model trained could also be used given that
# model parameters (e.g. embedding size and type) are the same.
MODEL_FILE_NAME = 'finalized_model.sav'
