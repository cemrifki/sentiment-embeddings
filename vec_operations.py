#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import numpy as np

import svd_U
from supervised_features import SupervisedFeatures


class VecOperations:
    """
    This class is used to perform basic vector operations, such as concatenation of embeddings
    and computing the average word vector of the words occurring in a review.
    """

    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels
        self.delta_idf_scores = SupervisedFeatures(revs, labels).extract_delta_idf_scores()

    def get_vecs(self):
        """
        This helper method returns embeddings of the ensemble form.
        In this ensemble form, corpus, lexical, and supervised features are combined.
        This and other corresponding "get_vecs" methods creates embeddings on a word-basis.

        :return: Combined embeddings for corpus words.
        :rtype: dict
        """
        return self.get_3_concatenated_vecs()

    def get_combination_of_3_approaches(self, diff_approach_vecs):  # corpus_svd_U + tdk_svd_U + 4_superv
        """
        This helper method combines three embeddings of different genres.

        :param diff_approach_vecs: The list of embeddings generated with different approaches.
        :type diff_approach_vecs: list
        :return: The combination of these embeddings.
        :rtype: dict
        """
        concatenated_vecs = {}

        all_words = self.get_all_words(diff_approach_vecs)

        for one_approach_vecs in diff_approach_vecs:

            one_approach_vecs_list = one_approach_vecs.keys()

            vec_dim_size = len(list(one_approach_vecs.values())[0])
            for word in all_words:

                if word not in concatenated_vecs:
                    concatenated_vecs[word] = []

                one_vec_component = one_approach_vecs[word] if word in one_approach_vecs_list \
                    else np.random.uniform(-0.1, 0.1, vec_dim_size)

                concatenated_vecs[word].extend(one_vec_component)

        return concatenated_vecs

    def get_3_concatenated_vecs(self):
        """
        A helper method that returns the concatenated vectors with respect to different types.

        :return: The combination of vectors.
        :rtype: dict
        """

        corpus_svd_u_vecs = svd_U.CorpusSvdU(self.revs, self.labels).get_corpus_svd_u_vecs()
        lexical_svd_u_vecs = svd_U.LexicalSvdU(self.revs, self.labels).get_lexical_svd_u_vecs()

        sf = SupervisedFeatures(self.revs, self.labels)
        self.delta_idf_scores = sf.delta_idf_scores
        four_delta_idf_vecs = sf.get_4_delta_idf_vecs()

        all_vecs = [corpus_svd_u_vecs, lexical_svd_u_vecs, four_delta_idf_vecs]

        return self.get_combination_of_3_approaches(all_vecs)

    def get_all_words(self, diff_approach_vecs):
        """
        This helper method obtains all words from embeddings dictionary.

        :param diff_approach_vecs: A list containing embeddings of different types.
        :type diff_approach_vecs: list
        :return: All words that are taken into account in this stage.
        :rtype: set
        """

        all_words = set([word for one_approach_vecs in diff_approach_vecs for word, vec in one_approach_vecs.items()])

        return all_words


def rev2avg_vec(rev, word_vecs):
    """
    This function takes the average of the words appearing in a review.
    This will later be used as an input feature in the classification stage.

    :param rev: A review.
    :type rev: list
    :param word_vecs: All word vectors generated based on corpora or lexicons or both.
    :type word_vecs: dict
    :return: Averaged vector representing a review.
    :rtype: list
    """

    vec_dim_size = len(list(word_vecs.values())[0])

    res_vec = np.random.uniform(-1.0, 1.0, vec_dim_size)
    for word in rev:
        vec = np.random.uniform(-1.0, 1.0, vec_dim_size) \
            if word not in word_vecs else np.array(word_vecs[word])
        res_vec = np.add(vec, res_vec)
    res_vec[(res_vec == np.inf) | (res_vec == -np.inf)] = 0
    res_vec = res_vec / len(rev)
    return res_vec.tolist()


def revs2avg_vecs(revs, word_vecs):
    """
    The list containing all the averaged embeddings of all the reviews in the corpus.

    :param revs: Corpus reviews.
    :type revs: list or ndarray
    :param word_vecs: All word vectors generated based on corpora or lexicons or both.
    :type word_vecs: dict
    :return: The averaged embeddings each representing a review.
    :rtype: list
    """

    return [rev2avg_vec(rev, word_vecs) for rev in revs]


if __name__ == "__main__":
    pass
