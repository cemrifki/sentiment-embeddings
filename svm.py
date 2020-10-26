#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import pickle

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score  # , f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

import constants
import emb_method_selector
import supervised_features
import vec_operations
from preprocessing import Preprocessing


def remove_nans_and_infs(data):
    """
    This function removes NaN and Inf numbers from the matrices.

    :param data: Embedding matrix.
    :type data: numpy.array
    :return: The matrix stripped off its unwanted numbers.
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame(data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[(df > 20) | (df < -20)] = 0
    return df


def run_cross_validation_svm(filename):
    """
    This function runs the support vector machines (SVM) algorithm using
        the cross-validation technique.

    :return: None. It just prints out the accuracy of the model.
    :rtype: None
    """

    fold_no = constants.CV_NUMBER
    kf = KFold(n_splits=fold_no)

    reviews, labels = Preprocessing().get_data(filename)

    tot_accuracy = 0.0

    iteration_no = 1
    for train, test in kf.split(reviews):

        print("Iteration #{}/{}:".format(iteration_no, fold_no))

        X_train, X_test, y_train, y_test = np.array(reviews)[train.astype(int)], np.array(reviews)[test.astype(int)], \
                                           np.array(labels)[train.astype(int)], np.array(labels)[test.astype(int)]

        model, sf, tr_vecs, imp = generate_model(X_train, y_train)

        y_pred = test_model(model, sf, tr_vecs, imp, X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print('Accuracy: {:.2f}%'.format(accuracy * 100))
        #        print("F1 score: " + str(f1_score(y_test, y_pred, average="macro")))
        #        print("Precision: " + str(precision_score(y_test, y_pred, average="macro")))
        #        print("Recall: " + str(recall_score(y_test, y_pred, average="macro")))

        tot_accuracy += accuracy

        iteration_no += 1
        print(" =" * 30)

    print("Average accuracy score: {:.2f}%".format(tot_accuracy / fold_no * 100))


def generate_model(reviews, labels):
    """
    This function generates the SVM model employing the training data only.

    :param reviews: Corpus reviews.
    :type reviews: list or ndarray
    :param labels: Sentiment labels that can be either positive or negative.
    :type labels: list or ndarray
    :return: clf (classifier), sf (supervised features), tr_vecs (training vectors), imp (imputer)
    :rtype: sklearn.svm._classes.SVC, supervised_features.SupervisedFeatures, dict, sklearn.impute._base.SimpleImputer
    """

    clf = svm.SVC(kernel='linear', C=1)

    X_train, y_train = np.array(reviews), np.array(labels)

    tr_vecs_class = emb_method_selector.embedding_methods[constants.EMBEDDING_TYPE]

    tr_obj = tr_vecs_class(X_train, y_train)
    tr_vecs = tr_obj.get_vecs()

    tr_avg_vecs = vec_operations.revs2avg_vecs(X_train, tr_vecs)

    if constants.USE_3_REV_POL_SCORES:  # Could be optimised.
        sf = supervised_features.SupervisedFeatures(X_train, y_train)
        tr_avg_vecs = sf.generate_revs_with_3_polarity_scores(X_train, tr_avg_vecs, y_train)
    else:
        sf = None

    tr_avg_vecs = remove_nans_and_infs(tr_avg_vecs)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(tr_avg_vecs)

    tr_avg_vecs = imp.transform(tr_avg_vecs)
    clf.fit(tr_avg_vecs, y_train)

    pickle.dump((clf, sf, tr_vecs, imp), open(constants.MODEL_FILE_NAME, 'wb'))

    print("Model has been trained.")

    return clf, sf, tr_vecs, imp


def test_model(model, sf, tr_vecs, imp, X_test):
    """
    This function predicts the labels of test data.
    It gets the data from the training model as input along with test data.

    :param model: SVM training model.
    :type model: sklearn.svm._classes.SVC
    :param sf: Supervised features generated from the model.
    :type sf: supervised_features.SupervisedFeatures
    :param tr_vecs: Vectors generated through the training data.
    :type tr_vecs: dict
    :param imp: Imputer used to handle specific values, such as NaNs and infinite values.
    :type imp: sklearn.impute._base.SimpleImputer
    :param X_test: Test data.
    :type X_test: list or ndarray
    :return: Predicted sentiment labels for the test data.
    :rtype: list
    """

    test_avg_vecs = vec_operations.revs2avg_vecs(X_test, tr_vecs)

    if constants.USE_3_REV_POL_SCORES:  # Could be optimised.
        test_avg_vecs = sf.generate_revs_with_3_polarity_scores(X_test, test_avg_vecs)

    test_avg_vecs = remove_nans_and_infs(test_avg_vecs)

    test_avg_vecs = imp.transform(test_avg_vecs)

    output = model.predict(test_avg_vecs)

    return output


def train_and_test_separate_files(train_path, test_path):
    """
    This function trains the model using the training data given in the path above (train_path).
    This also evaluates the test data specified in the path above (test_path).

    :param train_path: The path to the training data. This is located in the "input" folder.
    :type train_path: str
    :param test_path: The path to the test data. This also is located in the "input" folder.
    :type test_path: str
    :return: None. It just prints out the accuracy of the model.
    :rtype: None
    """

    tr_reviews, tr_labels = Preprocessing().get_data(train_path)
    X_train, y_train = np.array(tr_reviews), np.array(tr_labels)

    model, sf, tr_vecs, imp = generate_model(X_train, y_train)

    test_reviews, test_labels = Preprocessing().get_data(test_path)
    X_test, y_test = np.array(test_reviews), np.array(test_labels)

    y_pred = test_model(model, sf, tr_vecs, imp, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == "__main__":
    pass
