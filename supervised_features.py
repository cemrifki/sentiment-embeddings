#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import math
from collections import Counter
import random

import numpy as np

import preprocessing


class SupervisedFeatures:
    """
    This class exploits the sentiment characteristics that are present in
    the training data. Labels that are positive and negative are relied on.
    """
    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels
        self.delta_idf_scores = self.extract_delta_idf_scores()

    def get_vecs(self):
        """
        This helper method returns the minimum, maximum, mean, and self scores in accordance with
            contexts of target words. This is generated on a word-basis.

        :return: The four polarity scores of the context words of all corpus words.
        :rtype: dict
        """

        return self.get_4_delta_idf_vecs()

    def get_corp_count(self, revs):
        """
        This helper method computes the frequency of each word in the corpus.

        :param revs: All corpus reviews.
        :type revs: list
        :return: The frequencies of words in the corpus.
        :rtype: Counter
        """

        word_counts = Counter()
        for rev in revs:
            rev_set_words = set(rev)
            for word in rev_set_words:
                word_counts[word] += 1

        return word_counts

    def get_all_corpus_words(self, pos_revs, neg_revs):
        """
        This helper method returns all the words that appear in the whole of positive and negative corpora.

        :param pos_revs: Positive reviews.
        :type pos_revs: list
        :param neg_revs: Negative reviews.
        :type neg_revs: list
        :return: The set of all corpus words.
        :rtype: set
        """

        all_words = set()

        all_words = all_words.union(*pos_revs)
        all_words = all_words.union(*neg_revs)
        return all_words

    def get_all_revs(self, pos_revs, neg_revs):
        """
        This method combines all positive and negative reviews and returns the whole corpus.
        It is used in basic, supplementary operations.

        :param pos_revs: Positive reviews.
        :type pos_revs: list
        :param neg_revs: Negative reviews.
        :type neg_revs: list
        :return: The whole corpus.
        :rtype: list
        """

        all_revs = []
        all_revs.extend(pos_revs)
        all_revs.extend(neg_revs)
        return all_revs

    def extract_delta_idf_scores(self):
        """
        This method computes the delta-idf scores of words appearing in the training dataset.
        It should be noted that the only allowed sentiments are "P" (positive) and "N" (negative)
            in the dataset.

        :return: Words and their delta-idf scores.
        :rtype: dict
        """

        pos_indices = np.where(np.char.upper(self.labels.tolist()) == "P")
        neg_indices = np.where(np.char.upper(self.labels.tolist()) == "N")

        pos_revs = self.revs[pos_indices]
        neg_revs = self.revs[neg_indices]

        pos_counts = self.get_corp_count(pos_revs)
        neg_counts = self.get_corp_count(neg_revs)

        all_words = self.get_all_corpus_words(pos_revs, neg_revs)

        delta_idf_scores = {}
        for word in all_words:
            pos_val = 0
            neg_val = 0
            if word in pos_counts:
                pos_val = pos_counts[word]
            if word in neg_counts:
                neg_val = neg_counts[word]
            delta_idf_scores[word] = math.log((pos_val / len(pos_revs) + 0.001) /
                                              (neg_val / len(neg_revs) + 0.001))
        return delta_idf_scores

    def extract_context_delta_idf_scores(self, context_words, all_delta_idf_scores):
        """
        This helper method extracts the delta-idf scores of words (i.e. context) specified as in the input.

        :param context_words: A set of words.
        :type context_words: set or list
        :param all_delta_idf_scores: All delta-idf scores obtained by exploiting the label information.
        :type all_delta_idf_scores: dict
        :return: Delta-idf scores of the specified words appearing in context windows
        :rtype: list
        """

        return [all_delta_idf_scores[word] if word in all_delta_idf_scores else
                random.uniform(-1.0, 1.0) for word in context_words]

    def extract_all_context_delta_idf_scores(self, target_and_context_words, all_delta_idf_scores):
        """
        This helper method extracts the delta-idf scores of the words
            appearing in the same context windows as the target words.

        :param target_and_context_words: Target words and the words appearing in the same context windows
        :type target_and_context_words: dict
        :param all_delta_idf_scores: All delta-idf scores obtained by the training corpus.
        :type all_delta_idf_scores: dict
        :return: A dict that contains words as keys and all the delta-idf scores
            of words appearing in the same context windows.
        :rtype: dict
        """

        all_words_and_context_delta_idfs = {}
        for target_word, context_words in target_and_context_words.items():
            cont_delta_idfs = self.extract_context_delta_idf_scores(context_words, all_delta_idf_scores)
            all_words_and_context_delta_idfs[target_word] = cont_delta_idfs

        return all_words_and_context_delta_idfs

    def get_max_context_delta_idf_score(self, context_idf_scores):
        """
        This helper method obtains the maximal sentiment score from all the values.

        :param context_idf_scores: Delta-idf scores of the words appearing in context windows.
        :type context_idf_scores: dict
        :return: The maximum value of them all.
        :rtype: float
        """

        return max(context_idf_scores)

    def get_min_context_delta_idf_score(self, context_idf_scores):
        """
        This helper method obtains the minimal sentiment score from all the values.

        :param context_idf_scores: Delta-idf scores of the words appearing in context windows.
        :type context_idf_scores: dict
        :return: The minimal value of them all.
        :rtype: float
        """

        return min(context_idf_scores)

    def get_avg_context_delta_idf_score(self, context_idf_scores):
        """
        This helper method obtains the average sentiment score of all the values.

        :param context_idf_scores: Delta-idf scores of the words appearing in context windows.
        :type context_idf_scores: dict
        :return: The average value of them all.
        :rtype: float
        """

        return sum(context_idf_scores) / len(context_idf_scores)

    def get_context_4_delta_idf_scores(self, target_word, word_context_delta_idf_scores, delta_idf_scores):
        """
        This helper method generates a vector of length four on a word-basis.
        These values are maximal, minimal, averaged, and self- scores.

        :param target_word: The target word.
        :type target_word: str
        :param word_context_delta_idf_scores: All the delta-idf scores of words in the context windows.
        :type word_context_delta_idf_scores: dict
        :param delta_idf_scores: The delta-idf scores of words.
        :type delta_idf_scores: dict
        :return: 4-score vector.
        :rtype: list
        """

        supervised_4_scores = []
        three_scores = np.random.uniform(-1.0, 1.0, 3) if not word_context_delta_idf_scores else \
            ([self.get_max_context_delta_idf_score(word_context_delta_idf_scores)] +
             [self.get_min_context_delta_idf_score(word_context_delta_idf_scores)] +
             [self.get_avg_context_delta_idf_score(word_context_delta_idf_scores)])

        supervised_4_scores.extend(three_scores)
        supervised_4_scores.append(delta_idf_scores[target_word])

        return supervised_4_scores

    def get_all_context_4_delta_idf_scores(self, pos_revs, neg_revs):
        """
        This helper method generates four scores on a word-basis using positive and negative reviews.

        :param pos_revs: Positive reviews.
        :type pos_revs: list
        :param neg_revs: Negative reviews.
        :type neg_revs: list
        :return: A dictionary containing words as keys and four sentiment scores for each word as values.
        :rtype: dict
        """

        all_revs = self.get_all_revs(pos_revs, neg_revs)
        all_context_words = preprocessing.Preprocessing().get_all_context_words(all_revs=all_revs)

        context_delta_idf_scores = self.extract_all_context_delta_idf_scores(all_context_words, self.delta_idf_scores)

        all_context_4_delta_idf_scores = {}

        # A Pythonesque comprehension paradigm can also be used in lieu of the below for loop.
        for target_word, word_context_delta_idf_scores in context_delta_idf_scores.items():
            all_context_4_delta_idf_scores[target_word] = \
                self.get_context_4_delta_idf_scores(target_word,
                                                    word_context_delta_idf_scores,
                                                    self.delta_idf_scores)

        return all_context_4_delta_idf_scores

    def get_4_delta_idf_vecs(self):
        """
        This helper method gets all the 4-scores on a word-basis.

        :return: The dictionary including words as keys and their corresponding four sentiment scores as values.
        :rtype: dict
        """

        revs_np = np.array(self.revs)

        pos_indices = np.where(np.char.lower(self.labels.tolist()) == "p")
        neg_indices = np.where(np.char.lower(self.labels.tolist()) == "n")

        pos_revs = revs_np[pos_indices]
        neg_revs = revs_np[neg_indices]

        all_context_4_delta_idf_scores = self.get_all_context_4_delta_idf_scores(pos_revs, neg_revs)
        return all_context_4_delta_idf_scores

    def get_review_polarity_scores(self, rev, delta_idf_scores):
        """
        This helper method returns all the delta-idf scores of a review.

        :param rev: A review.
        :type rev: list
        :param delta_idf_scores: All delta-idf scores of words in the corpus.
        :type delta_idf_scores: dict
        :return: All delta-idf scores of a review.
        :rtype: list
        """

        rev_delta_idf_scores = [delta_idf_scores[word] for word in rev if word in delta_idf_scores]
        return rev_delta_idf_scores

    def get_review_3_polarity_scores(self, rev, delta_idf_scores):
        """
        This helper method returns three delta-idf scores (minimum, maximum, and mean) per review.

        :param rev: A review.
        :type rev: list
        :param delta_idf_scores: All delta-idf scores generated in accordance with the training set.
        :type delta_idf_scores: dict
        :return: The three polarity scores.
        :rtype: list
        """

        review_polarity_scores = self.get_review_polarity_scores(rev, delta_idf_scores)

        if len(review_polarity_scores) == 0:
            return np.random.uniform(-1.0, 1.0, 3)  # np.random.rand(1).tolist()

        min_pol = max(review_polarity_scores)
        max_pol = min(review_polarity_scores)
        mean_pol = sum(review_polarity_scores) / len(review_polarity_scores)

        review_3_polarity_scores = list([])

        review_3_polarity_scores.append(min_pol)
        review_3_polarity_scores.append(max_pol)
        review_3_polarity_scores.append(mean_pol)

        return review_3_polarity_scores

    def get_all_revs_3_polarity_scores(self, revs, *labels):
        """
        This helper method returns minimum, msximum, and mean polarity scores for each review.

        :param revs: All corpus reviews.
        :type revs: list
        :param labels: Polairities that can be either positive or negative.
        :type labels: list
        :return: Reviews and their corresponding three polarities (extracted on a review-basis).
        :rtype: list
        """

        return [self.get_review_3_polarity_scores(rev, self.delta_idf_scores) for rev in revs]

    def generate_revs_with_3_polarity_scores(self, revs, avg_vecs_matr, *labels):
        """
        This helper method returns three polarities (delta-idf scores) for each review
            and concatenates them with other embeddings of the words in the same review.

        :param revs: Reviews.
        :type revs: list or ndarray
        :param avg_vecs_matr: The averaged embeddings of the reviews.
        :type avg_vecs_matr: list
        :param labels: Polarities of reviews.
        :type labels: list or ndarray
        :return: Embeddings concatenated with three polarity scores on a review-basis.
        :rtype: np.ndarray
        """

        revs_three_pol_scores = self.get_all_revs_3_polarity_scores(revs, *labels)
        return np.hstack((avg_vecs_matr, revs_three_pol_scores))
