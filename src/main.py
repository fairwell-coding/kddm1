import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from helper import unpack_dataset
from helper import read_xls_file
import logging

from src.preprocessing import outlier_detection, train_test_split, preprocess_data, get_evaluation_data, RANDOM_STATE

USE_NMF = True


def main():
    logging.basicConfig(level=logging.INFO)

    unpack_dataset("jester_dataset_1_1.zip")
    unpack_dataset("jester_dataset_1_2.zip")
    unpack_dataset("jester_dataset_1_3.zip")

    _, jester_1 = read_xls_file("jester-data-1.xls")
    # _, jester_2 = read_xls_file("jester-data-2.xls")
    # _, jester_3 = read_xls_file("jester-data-3.xls")

    data_preprocessed = preprocess_data(jester_1, USE_NMF)
    outlier_detection(data_preprocessed)
    test_data, train_data = train_test_split(data_preprocessed, USE_NMF)
    H, W = __nmf_scikit_learn(train_data)
    M_hat = np.matmul(W, H)
    y_hat = get_evaluation_data(M_hat)
    rmse = __evaluate_nmf_using_rmse(y_hat, test_data)

    print('rmse: ', rmse)


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


if __name__ == '__main__':
    main()
