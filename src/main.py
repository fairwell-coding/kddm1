import logging

import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error

from helper import read_xls_file
from helper import unpack_dataset
from src.preprocessing import train_test_split, preprocess_data, get_evaluation_data, RANDOM_STATE, \
    prepare_data
from src.scikit_learn_nan import evaluate_nmf_using_rmse_with_nans

PREPROCESS_DATA = True

# Note from Flo:
# in case we really wanna clean our data and remove outliers
# don't think we wanna do that, because the user data where a suer 'voted against the mass' is quite valuable
# for our algorithm, should break up the trend a bit, otherwise we might just add an additional bias by doing so
# especially since both algorithm classify many points a outliers:
# local outlier factor found 16515 outliers from 24982 data points
# got 5110 outliers and 19872 correct samples in the given data according to the IsolationForest classifier
REMOVE_OUTLIERS = False


def main():
    logging.basicConfig(level=logging.INFO)

    # unpack_dataset("jester_dataset_1_1.zip")
    # unpack_dataset("jester_dataset_1_2.zip")
    # unpack_dataset("jester_dataset_1_3.zip")

    _, jester_1 = read_xls_file("jester-data-1.xls")
    # _, jester_2 = read_xls_file("jester-data-2.xls")
    # _, jester_3 = read_xls_file("jester-data-3.xls")

    #__svd_nmf_with_no_imputation(jester_1)

    # !!! data_prepared contains NaNs !!!
    data_prepared = prepare_data(jester_1, False)

    # if outliers should be removed, use return value of outlier_detection
    # and only use 1 of the 2 techniques outlier_detection contains both
    # Flo: removed outlier detection from workflow, because we not gonna use it anyway
    # and algorithms cant work with NaNs
    # outlier_detection(data_prepared, REMOVE_OUTLIERS)

    rmse = process_data_with_nmf(data_prepared)
    print('rmse for NMF:', rmse)

    rmse = process_data_with_svd(data_prepared)
    print('rmse for SVD: ', rmse)

    # process_data_with_light_gcn(data_prepared)


def process_data_with_nmf(dataset):

    if PREPROCESS_DATA is True:
        data_preprocessed, nan_mask = preprocess_data(dataset)
    else:
        # just replace NaNs and move data to positive interval
        _, nan_mask = preprocess_data(dataset)
        data_preprocessed = np.nan_to_num(dataset, nan=0.0)
        data_preprocessed = data_preprocessed + 10

    test_data, train_data = train_test_split(data_preprocessed, fill_test_data_with_zero=True)
    H, W = __nmf_scikit_learn(train_data)
    M_hat = np.matmul(W, H)
    y_hat = get_evaluation_data(M_hat)
    y_hat = np.where(nan_mask, y_hat, np.nan)
    test_data = np.where(nan_mask, test_data, np.nan)
    rmse = evaluate_nmf_using_rmse_with_nans(y_hat, test_data)
    return rmse


def process_data_with_svd(dataset):
    print("SVD started")
    if PREPROCESS_DATA is True:
        data_preprocessed, nan_mask = preprocess_data(dataset, scale_to_positive_interval=False)
    else:
        # just replace NaNs with 0
        _, nan_mask = preprocess_data(dataset)
        data_preprocessed = np.nan_to_num(dataset, nan=0.0)

    test_data, train_data = train_test_split(data_preprocessed, fill_test_data_with_zero=True)
    H, W = __svd_scikit_learn(train_data)
    M_hat = np.matmul(W, H)
    y_hat = get_evaluation_data(M_hat)
    y_hat = np.where(nan_mask, y_hat, np.nan)
    test_data = np.where(nan_mask, test_data, np.nan)
    rmse = evaluate_nmf_using_rmse_with_nans(y_hat, test_data)
    return rmse


def __svd_scikit_learn(train_data):
    latent_space_dimensions = 50
    num_epochs = 10

    svd = TruncatedSVD(random_state=RANDOM_STATE, n_components=latent_space_dimensions, n_iter=num_epochs)
    W = svd.fit_transform(train_data)
    H = svd.components_

    return H, W


def __nmf_scikit_learn(train_data):
    latent_space_dimensions = 50
    num_epochs = 100
    init_method = 'nndsvd'

    nmf = NMF(random_state=RANDOM_STATE, init=init_method, n_components=latent_space_dimensions, max_iter=num_epochs)
    W = nmf.fit_transform(train_data)
    H = nmf.components_

    return H, W


def __evaluate_nmf_using_rmse(y_hat, test_data):
    rmse = mean_squared_error(test_data, y_hat, squared=False)
    return rmse


if __name__ == '__main__':
    main()
