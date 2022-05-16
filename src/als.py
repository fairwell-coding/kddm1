import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from src.helper import replace_nans
from src.preprocessing import train_test_split

NUM_TEST_COLUMNS = 10
NUM_TEST_ROWS = 1000


class AlternatingLeastSquares:
    def __init__(self, train, test, num_features=20, num_iterations=100, learning_rate=0.001):
        self.num_users, self.num_items = train.shape
        self.train_set = train
        self.test_set = test
        self.num_features = num_features
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.U = np.random.rand(self.num_users, num_features)
        self.V = np.random.rand(self.num_items, num_features)
        self.test_mse_record = []
        self.train_mse_record = []

    def train(self):
        for i in range(self.num_iterations):
            self.U = self.calc_step(np.nan_to_num(self.train_set, 0), self.V)
            self.V = self.calc_step(np.nan_to_num(self.train_set.T, 0), self.U)
            pred_test, pred_train = train_test_split(self.predict())
            test_mse = self.compute_mse(self.test_set, pred_test)
            train_mse = self.compute_mse(self.train_set, pred_train)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)

    def calc_step(self, M, fixed):
        inv = np.linalg.inv(fixed.T.dot(fixed))
        return M.dot(fixed).dot(inv)

    def predict(self):
        return self.U.dot(self.V.T)

    def compute_mse(self, y_true, y_pred):
        mask = np.isnan(y_true)
        y_true = y_true[~mask]
        y_pred = y_pred[~mask]
        mse = mean_squared_error(y_true, y_pred, squared=False)
        return mse

    def plot_learning_curve(self):
        """visualize the training/testing loss"""
        linewidth = 3
        plt.plot(self.test_mse_record, label='Test', linewidth=linewidth)
        plt.plot(self.train_mse_record, label='Train', linewidth=linewidth)
        plt.xlabel('iterations')
        plt.ylabel('MSE')
        plt.legend(loc='best')
        plt.show()
