import numpy as np

class DataSimulate:
    def __init__(self, rand_features = 5, data_points = 1000):
        self.data_points = data_points
        self.clean_points = np.random.binomial(1, 0.5, (self.data_points, 3))
        self.labels = np.dot(self.clean_points, np.array([1,2,4]))
        self.points = self.clean_points + np.random.normal(0, 0.1, (self.data_points, 3))
        self.dataset = np.random.rand(self.data_points, 10 + rand_features)
        for i in range(self.data_points):
            offset = self.labels[i];
            for j in range(3):
                self.dataset[i, offset + j] = self.points[i, j]
        self.labels = np.eye(8)[self.labels]
        self.index = 0

    def next_batch(self, batch_size):

        new_index = (self.index + batch_size) % self.data_points
        if self.index + batch_size <= self.data_points:
            result = [self.dataset[self.index:self.index+batch_size], self.labels[self.index:self.index+batch_size]]
        else:
            dataset = self.dataset[self.index:] + self.dataset[:new_index]
            labels = self.labels[self.index:] + self.labels[:new_index]
            result = [dataset, labels]
        self.index = new_index
        return result
