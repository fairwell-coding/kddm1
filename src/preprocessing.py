import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from src.plots import plot_local_outlier_factor, plot_individual_joke_rating, plot_joke_rating, plot_qq_individual_joke

RANDOM_STATE = 42
NUM_TEST_COLUMNS = 10
NUM_TEST_ROWS = 1000


def outlier_detection(preprocessed_data):
    """
    Handles the outlier detection

    methods used:

    * local outlier factor
    * IsolationForest

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
    print("got", no_outliers, "outliers and", no_correct_samples,
          "correct samples in the given data according to the IsolationForest classifier")


def train_test_split(data_preprocessed, use_nmf=True):
    top = data_preprocessed[:-NUM_TEST_ROWS]  # All rows - Test_rows
    bottom_left = data_preprocessed[-NUM_TEST_ROWS:, :-NUM_TEST_COLUMNS]
    # if nmf is used fill now empty values with 0
    if use_nmf is True:
        bottom_right = np.full((NUM_TEST_ROWS, NUM_TEST_COLUMNS), 0)
    else:
        bottom_right = np.full((NUM_TEST_ROWS, NUM_TEST_COLUMNS), np.nan)

    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    train_data = np.concatenate((top, bottom), axis=0)

    test_data = data_preprocessed[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]

    return test_data, train_data


def __normalize_dataset(dataset):
    mean = np.nanmean(dataset, axis=1)[:, None]  # Mean joke rating per user
    std = np.nanstd(dataset, axis=1)[:, None]  # standard deviation per user
    dataset = (np.array(dataset) - mean) / std  # normalize
    return dataset


def preprocess_data(dataset, use_nmf=True):
    dataset = dataset[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    dataset = dataset.astype(float)  # convert joke ratings to float32
    dataset = np.where(dataset == 99, np.nan, dataset)  # set not-rated-jokes to NaN

    # dataset = __normalize_dataset(dataset)
    # dataset += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # dataset += 10  # alternative: shift all joke ratings into positive number range

    if use_nmf is True:
        dataset = dataset - np.nanmean(dataset)  # center by mean
        dataset = np.nan_to_num(dataset, nan=0.0)  # NaNs are now average rating
        plot_individual_joke_rating(dataset[:, 40], '(40 after centering)')
        plot_joke_rating(dataset, ' after centering')
        min_value = np.nanmin(dataset)
        # after centering the data by mean, some values might be out of the interval
        # however nmf only accepts positive values, so we transform everything to [0,20]
        print('min value(', min_value, ') in data after centering rating is out of interval, scaling interval')
        scaler = MinMaxScaler(feature_range=(0, 20))
        dataset = scaler.fit_transform(dataset)

    plot_individual_joke_rating(dataset[:, 40], 40)
    plot_qq_individual_joke(dataset[:, 40], 40)
    plot_joke_rating(dataset)

    return dataset


def get_evaluation_data(M_hat):
    return M_hat[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]
