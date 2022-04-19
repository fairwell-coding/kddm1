import numpy as np

from src.helper import replace_nans


class AlternatingLeastSquares:
    def __init__(self, M, num_features=10, num_iterations=10, learning_rate=0.001):
        self.num_users, self.num_items = M.shape
        self.M = replace_nans(M, 0)
        self.num_features = num_features
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.U = np.random.rand(self.num_users, num_features)
        self.V = np.random.rand(self.num_items, num_features)

    def train(self):
        for i in range(self.num_iterations):
            self.U = self.als_step(self.M, self.V)
            self.V = self.als_step(self.M.T, self.U)

    def als_step(self, M, fixed):
        inv = np.linalg.inv(fixed.T.dot(fixed))
        return M.dot(fixed).dot(inv)
