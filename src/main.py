import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor

from helper import unpack_dataset
from helper import read_xls_file
from src.plots import plot_joke_rating, plot_individual_joke_rating, plot_qq_individual_joke, plot_local_outlier_factor
import logging

NUM_TEST_COLUMNS = 10
NUM_TEST_ROWS = 1000

RANDOM_STATE = 42


def main():
    logging.basicConfig(level=logging.DEBUG)

    unpack_dataset("jester_dataset_1_1.zip")
    unpack_dataset("jester_dataset_1_2.zip")
    unpack_dataset("jester_dataset_1_3.zip")

    _, jester_1 = read_xls_file("jester-data-1.xls")
    # _, jester_2 = read_xls_file("jester-data-2.xls")
    # _, jester_3 = read_xls_file("jester-data-3.xls")

    data_preprocessed = __preprocess_data(jester_1)
    test_data, train_data = __train_test_split(data_preprocessed)
    H, W = __nmf_scikit_learn(train_data)
    M_hat = np.matmul(W, H)
    y_hat = M_hat[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]
    rmse = __evaluate_nmf_using_rmse(y_hat, test_data)

    print('rmse: ', rmse)


def __outlier_detection(preprocessed_data):
    """
    Handles the outlier detection

    methods used:

    * local outlier factor

    :param preprocessed_data: data for outlier detection, must not contain NaN
    """
    __isolation_forest_outlier(preprocessed_data)
    __local_outlier_factor(preprocessed_data)


def __local_outlier_factor(preprocessed_data):
    """
    Plots the local outlier factor from the given data

    :param preprocessed_data: data for outlier detection, must not contain NaN
    """
    n_neighbors = 20  # default value
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    clf.fit_predict(preprocessed_data)
    data_scores = clf.negative_outlier_factor_
    plot_local_outlier_factor(preprocessed_data, data_scores)


def __isolation_forest_outlier(preprocessed_data):
    """
    Prints how many outliers the IsolationForest classifier found in the given dataset

    :param preprocessed_data:
    """
    clf = IsolationForest(random_state=RANDOM_STATE)
    data_score = clf.fit_predict(preprocessed_data)

    no_outliers = np.count_nonzero(data_score == -1)
    no_correct_samples = np.count_nonzero(data_score == 1)
    print("got", no_outliers, " outliers and", no_correct_samples,
          " correct samples in the given data according to the IsolationForest classifier")


def __nmf_scikit_learn(train_data):
    latent_space_dimensions = 50
    num_epochs = 2000
    init_method = 'nndsvd'

    nmf = NMF(random_state=RANDOM_STATE, init=init_method, n_components=latent_space_dimensions, max_iter=num_epochs)
    W = nmf.fit_transform(train_data)
    H = nmf.components_

    return H, W


def __evaluate_nmf_using_rmse(y_hat, test_data):
    rmse = mean_squared_error(test_data, y_hat, squared=False)
    return rmse


def __train_test_split(data_preprocessed):
    top = data_preprocessed[:-NUM_TEST_ROWS]  # All rows - Test_rows
    bottom_left = data_preprocessed[-NUM_TEST_ROWS:, :-NUM_TEST_COLUMNS]
    bottom_right = np.full((NUM_TEST_ROWS, NUM_TEST_COLUMNS), np.nan)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    train_data = np.concatenate((top, bottom), axis=0)

    test_data = data_preprocessed[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]

    return test_data, train_data


def __normalize_dataset(dataset):
    mean = np.nanmean(dataset, axis=1)[:, None]  # Mean joke rating per user
    std = np.nanstd(dataset, axis=1)[:, None]  # standard deviation per user
    dataset = (np.array(dataset) - mean)/std  # normalize
    return dataset


def __preprocess_data(dataset):
    dataset = dataset[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    dataset = dataset.astype(float)  # convert joke ratings to float32
    dataset = np.where(dataset == 99, np.nan, dataset)  # set not-rated-jokes to NaN

    plot_individual_joke_rating(dataset[:,40], 40)
    plot_qq_individual_joke(dataset[:,40], 40)
    plot_joke_rating(dataset)

    dataset = __normalize_dataset(dataset)
    # dataset += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # dataset += 10  # alternative: shift all joke ratings into positive number range
    return dataset


if __name__ == '__main__':
    main()
