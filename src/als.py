import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from src.helper import replace_nans

NUM_TEST_COLUMNS = 10
NUM_TEST_ROWS = 1000


class AlternatingLeastSquares:
    def __init__(self, train, test, num_features=10, num_iterations=10, learning_rate=0.001):
        self.num_users, self.num_items = train.shape
        self.train_set = replace_nans(train, 0)
        self.test_set = replace_nans(test, 0)
        self.num_features = num_features
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.U = np.random.rand(self.num_users, num_features)
        self.V = np.random.rand(self.num_items, num_features)
        self.test_mse_record = []
        self.train_mse_record = []

    def train(self):
        for i in range(self.num_iterations):
            self.U = self.calc_step(self.train_set, self.V)
            self.V = self.calc_step(self.train_set.T, self.U)
            pred_test, pred_train = self.split(self.predict())
            test_mse = self.compute_mse(self.test_set, pred_test)
            train_mse = self.compute_mse(self.train_set, pred_train)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)

    def calc_step(self, M, fixed):
        inv = np.linalg.inv(fixed.T.dot(fixed))
        return M.dot(fixed).dot(inv)

    def predict(self):
        return self.U.dot(self.V.T)

    def split(self, matrix):
        top = matrix[:-NUM_TEST_ROWS]  # All rows - Test_rows
        bottom_left = matrix[-NUM_TEST_ROWS:, :-NUM_TEST_COLUMNS]
        bottom_right = np.full((NUM_TEST_ROWS, NUM_TEST_COLUMNS), np.nan)
        bottom = np.concatenate((bottom_left, bottom_right), axis=1)
        train_data = np.concatenate((top, bottom), axis=0)
        test_data = matrix[-NUM_TEST_ROWS:, -NUM_TEST_COLUMNS:]
        return test_data, train_data

    def compute_mse(self, y_true, y_pred):
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask], squared=False)
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