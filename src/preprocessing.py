import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from src.plots import plot_local_outlier_factor, plot_individual_joke_rating, plot_joke_rating, plot_qq_individual_joke, \
    boxplot_rating_distribution, distogram_rating_distribution

RANDOM_STATE = 42
NUM_TEST_COLUMNS = 10
NUM_TEST_ROWS = 1000


def outlier_detection(preprocessed_data, remove_outliers=False):
    """
    Handles the outlier detection

    methods used:

    * local outlier factor
    * IsolationForest

    :param remove_outliers: if outliers should be removed
    :param preprocessed_data: data for outlier detection, must not contain NaN

    :return eiter the original data or the data with removed outliers
    """
    isolation_forest_outlier(preprocessed_data, remove_outliers)
    local_outlier_factor(preprocessed_data, remove_outliers)


def local_outlier_factor(preprocessed_data, remove_outliers=False):
    """
    Plots the local outlier factor from the given data

    :param remove_outliers: if outliers should be removed
    :param preprocessed_data: data for outlier detection, must not contain NaN

    :return eiter the original data or the data with removed outliers
    """
    n_neighbors = 20  # default value
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    is_inlier = clf.fit_predict(preprocessed_data)
    data_scores = clf.negative_outlier_factor_
    plot_local_outlier_factor(preprocessed_data, data_scores)

    print('local outlier factor found', np.count_nonzero(is_inlier == -1), 'outliers from', np.count_nonzero(is_inlier), 'data points')
    if remove_outliers is True:
        outlier_index = np.where(is_inlier == -1)
        cleaned_data = np.delete(preprocessed_data, outlier_index, axis=0)
        return cleaned_data

    return preprocessed_data


def isolation_forest_outlier(preprocessed_data, remove_outliers=False):
    """
    Prints how many outliers the IsolationForest classifier found in the given dataset

    :param remove_outliers: if the outliers should be removed
    :param preprocessed_data: data for outlier detection, must not contain NaN
    """
    clf = IsolationForest(random_state=RANDOM_STATE)
    data_score = clf.fit_predict(preprocessed_data)

    no_outliers = np.count_nonzero(data_score == -1)
    no_correct_samples = np.count_nonzero(data_score == 1)
    print("got", no_outliers, "outliers and", no_correct_samples,
          "correct samples in the given data according to the IsolationForest classifier")

    if remove_outliers is True:
        outlier_index = np.where(data_score == -1)
        cleaned_data = np.delete(preprocessed_data, outlier_index, axis=0)
        return cleaned_data

    return preprocessed_data


def train_test_split(data_preprocessed, fill_test_data_with_zero=True):
    top = data_preprocessed[:-NUM_TEST_ROWS]  # All rows - Test_rows
    bottom_left = data_preprocessed[-NUM_TEST_ROWS:, :-NUM_TEST_COLUMNS]
    # if nmf is used fill now empty values with 0
    if fill_test_data_with_zero is True:
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


def prepare_data(dataset):
    dataset = dataset[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    dataset = dataset.astype(float)  # convert joke ratings to float32
    dataset = np.where(dataset == 99, np.nan, dataset)  # set not-rated-jokes to NaN

    boxplot_rating_distribution(dataset)
    distogram_rating_distribution(dataset)

    return dataset


def preprocess_data(dataset):
    # dataset = __normalize_dataset(dataset)
    # dataset += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # dataset += 10  # alternative: shift all joke ratings into positive number range

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
