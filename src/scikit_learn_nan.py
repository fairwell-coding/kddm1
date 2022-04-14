from sklearn.decomposition import NMF

RANDOM_STATE = 42


def nmf_scikit_learn_with_nans(train_data):
    latent_space_dimensions = 50
    num_epochs = 2000
    init_method = 'random'
    solver = 'mu'

    nmf = NMF(random_state=RANDOM_STATE, init=init_method, n_components=latent_space_dimensions, max_iter=num_epochs,
              solver=solver)
    W = nmf.fit_transform(train_data)
    H = nmf.components_

    return H, W
