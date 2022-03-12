import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

from helper import unpack_dataset
from helper import read_xls_file
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

    data_proprocessed = __preprocess_data(jester_1)
    test_data, train_data = __train_test_split(data_proprocessed)
    H, W = __nmf_scikit_learn(train_data)
    rmse = __evaluate_nmf_using_rmse(H, W, test_data)

    print('rmse: ', rmse)


def __nmf_scikit_learn(train_data):
    latent_space_dimensions = 50
    num_epochs = 2000
    init_method = 'nndsvd'

    nmf = NMF(random_state=RANDOM_STATE, init=init_method, n_components=latent_space_dimensions, max_iter=num_epochs)
    W = nmf.fit_transform(train_data)
    H = nmf.components_

    return H, W


def __evaluate_nmf_using_rmse(H, W, test_data):
    M_hat = np.matmul(W, H)
    y_hat = M_hat[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]
    rmse = mean_squared_error(test_data, y_hat, squared=False)

    return rmse


def __train_test_split(data_proprocessed):
    top = data_proprocessed[:-NUM_TEST_ROWS]
    bottom_left = data_proprocessed[-NUM_TEST_ROWS:, :-NUM_TEST_COLUMNS]
    bottom_right = np.full((NUM_TEST_ROWS, NUM_TEST_COLUMNS), np.nan)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    train_data = np.concatenate((top, bottom), axis=0)

    test_data = data_proprocessed[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]

    return test_data, train_data


def __preprocess_data(dataset):
    dataset = dataset[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    dataset = dataset.astype(float)  # convert joke ratings to float32
    dataset = np.where(dataset == 99, np.nan, dataset)  # set not-rated-jokes to NaN
    dataset = normalize(dataset)  # normalize over feature dimensions (here not necessary)
    dataset += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # dataset += 10  # alternative: shift all joke ratings into positive number range

    return dataset


if __name__ == '__main__':
    main()
