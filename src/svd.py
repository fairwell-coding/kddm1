import numpy as np
import surprise  # run 'pip install scikit-surprise' to install surprise
from surprise import Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd


def __svd(dataset):
    print("--------------------------------------------------")
    print("    START SVD with stochastic gradient descent    ")
    print("--------------------------------------------------")

    dataset = pd.DataFrame(dataset[1:, 1:])
    data = dataset.astype(float)

    ratings_dict = {'jokeID': [],
                    'userID': [],
                    'rating': []}
    print("[Start] Converting data to Dataframe")
    for user_id, user_vector in data.iterrows():
        for joke_id, rating in enumerate(user_vector):
            if rating >= 20.0:
                continue
            ratings_dict['jokeID'].append(joke_id)
            ratings_dict['userID'].append(user_id)
            ratings_dict['rating'].append(rating)

    reader = Reader(rating_scale=(1, 1810381))
    df = pd.DataFrame(ratings_dict)
    data = surprise.Dataset.load_from_df(df[['userID', 'jokeID', 'rating']], reader)
    print("[END] Converting data to Dataframe")

    epochs = 10
    train_set, validation_set = train_test_split(data, shuffle=True, test_size=.25)

    algo = SVD(verbose=True, n_epochs=epochs)
    print("[Start] Fit SVD")
    algo.fit(train_set)
    print("[END] Fit SVD")
    predictions = algo.test(validation_set)
    print(f"RMSE for SVD with stochastic gradient descent for {epochs} epochs")
    accuracy.rmse(predictions, verbose=True)

    print("--------------------------------------------------")
    print("     END SVD with stochastic gradient descent     ")
    print("--------------------------------------------------")
