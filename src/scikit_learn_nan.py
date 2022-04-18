import numpy as np
from sklearn.decomposition import NMF

RANDOM_STATE = 42


def nmf_scikit_learn_with_nans(train_data):
    # not every init_method and solver works with the nan values in the nmf implementation
    # params below do accept nan as an input
    latent_space_dimensions = 50
    num_epochs = 2000
    init_method = 'random'
    solver = 'mu'

    nmf = NMF(random_state=RANDOM_STATE, init=init_method, n_components=latent_space_dimensions, max_iter=num_epochs,
              solver=solver)
    W = nmf.fit_transform(train_data)
    H = nmf.components_

    return H, W


def evaluate_nmf_using_rmse_with_nans(y_hat, test_data):
    """
    calculates the rmse based on y_hat and test_data

    nan values are going to be ignored

    :return: rmse
    """
    rmse = np.sqrt(np.nanmean((test_data-y_hat)**2))
    return rmse