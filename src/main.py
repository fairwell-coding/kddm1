import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from helper import unpack_dataset
from helper import read_xls_file
import logging


RANDOM_STATE = 42


def main():
    logging.basicConfig(level=logging.DEBUG)

    unpack_dataset("jester_dataset_1_1.zip")
    unpack_dataset("jester_dataset_1_2.zip")
    unpack_dataset("jester_dataset_1_3.zip")

    _, jester_1 = read_xls_file("jester-data-1.xls")
    # _, jester_2 = read_xls_file("jester-data-2.xls")
    # _, jester_3 = read_xls_file("jester-data-3.xls")

    jester_1_preprocessed = __preprocess_data(jester_1)

    nmf = NMF(random_state=RANDOM_STATE, init='random', n_components=50, max_iter=500)
    W = nmf.fit_transform(jester_1_preprocessed)
    H = nmf.components_

    print('x')


def __preprocess_data(dataset):
    dataset = dataset[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    dataset = dataset.astype(float)  # convert joke ratings to float32
    np.where(dataset == 99, np.nan, dataset)  # set not-rated-jokes to NaN
    dataset = normalize(dataset)  # normalize over feature dimensions (here not necessary)
    dataset += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # dataset += 10  # alternative: shift all joke ratings into positive number range

    return dataset


if __name__ == '__main__':
    main()
