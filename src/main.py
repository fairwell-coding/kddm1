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
    # unpack_dataset("jester_dataset_1_2.zip")
    # unpack_dataset("jester_dataset_1_3.zip")

    _, jester_1 = read_xls_file("jester-data-1.xls")
    # jester_2 = read_xls_file("jester-data-2.xls")
    # jester_3 = read_xls_file("jester-data-3.xls")

    jester_1 = __prepare_data(jester_1)

    nmf = NMF(random_state=RANDOM_STATE, init='random', n_components=50, max_iter=500)
    W = nmf.fit_transform(jester_1)
    H = nmf.components_

    print('x')


def __prepare_data(jester_1):
    jester_1 = jester_1[1:, 1:]  # data cleaning: 1st sample has several invalid data inputs, e.g. '8.5.1'
    jester_1 = jester_1.astype(float)  # convert joke ratings to float32
    np.where(jester_1 == 99, np.nan, jester_1)  # set not-rated-jokes to NaN
    jester_1 = normalize(jester_1)  # normalize over feature dimensions (here not necessary)
    jester_1 += 1  # normalized data lies within interval [-1, 1], hence we shift the data to the compact interval [0, 2]
    # jester_1 += 10  # alternative: shift all joke ratings into positive number range

    return jester_1


if __name__ == '__main__':
    main()
