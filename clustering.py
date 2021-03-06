#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
========================
Fuzzy c-means clustering
========================

Fuzzy logic principles can be used to cluster multidimensional data, assigning
each point a *membership* in each cluster center from 0 to 100 percent. This
can be very powerful compared to traditional hard-thresholded clustering where
every point is assigned a crisp, exact label.

Fuzzy c-means clustering is accomplished via ``skfuzzy.cmeans``, and the
output from this function can be repurposed to classify new data according to
the calculated clusters (also known as *prediction*) via
``skfuzzy.cmeans_predict``

Data generation and setup
-------------------------

In this example we will first undertake necessary imports, then define some
test data to work with.

"""

import skfuzzy as fuzz

import PMI_matrix
import constants


class Clustering:
    """
    This class is used to perform c-means clustering and generate embeddings
        per word based on this clustering approach.
    """

    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels

    def get_vecs(self):
        """
        This helper interface method returns vectors generated by the clustering method.

        :return: A dictionary of keys as words and values as the above-mentioned embeddings.
        :rtype: dict
        """
        return self.get_training_cluster_vecs()

    def get_training_cluster_vecs(self):
        """
        This function generates embeddings based on the clustering approach.
        C-mean clustering method is employed on the PMI matrix of the training corpus.

        :return: A dictionary of words as keys and vectors generated by the clustering model as values.
        :rtype: dict
        """

        pmi_dict_keys, pmi_matr = PMI_matrix.prepare_pmi(self.revs)
        no_of_clusters = min(constants.EMBEDDING_SIZE, len(pmi_matr[0]))

        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            pmi_matr, no_of_clusters, 2, error=0.005, maxiter=8)

        res_dict = dict(zip(pmi_dict_keys, u_orig.T))
        return res_dict
