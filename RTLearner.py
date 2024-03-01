import numpy as np
from scipy import stats

class RTLearner(object) :
    def __init__(self, leaf_size=1, verbose=False):
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.node = None
        self.tree = None


    def author(self):
        return "ashabou3"

    def add_evidence(self, train_x, train_y):
        if self.verbose:
            pass
        self.tree = self.build_tree(train_x, train_y)

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            y = data_y[0]
            return np.array([[-1, y, -1, -1]])
        else:
            xi = np.random.randint(0, data_x.shape[1])
            split_val = np.median(data_x[:, xi])
            unique_target_values = np.unique(data_x[:, xi] <= split_val)
            if len(unique_target_values) == 1:
                y = stats.mode(data_y[(data_x[:, xi] <= split_val)])
                return np.array([[-1, 0.0 if len(y) > 1 else y[0], -1, -1]])
            x_left_data = data_x[data_x[:, xi] <= split_val]
            y_left_data = data_y[data_x[:, xi] <= split_val]
            x_right_data = data_x[data_x[:, xi] > split_val]
            y_right_data = data_y[data_x[:, xi] > split_val]
            left_tree = self.build_tree(x_left_data, y_left_data)
            right_tree = self.build_tree(x_right_data, y_right_data)
            root = np.array([[xi, split_val, 1, len(left_tree) + 1]])
            return np.concatenate([root, left_tree, right_tree])

    def traverse_tree_and_get_value(self, point, level):
        if self.tree[level][2] == -1 and self.tree[level][3] == -1:
            return self.tree[level][1]
        if point[int(self.tree[level][0])] <= self.tree[level][1]:
            return self.traverse_tree_and_get_value(point, level + int(self.tree[level][2]))
        else:
            return self.traverse_tree_and_get_value(point, level + int(self.tree[level][3]))
    def query(self, points):
        num_of_x = points.shape[0]
        predicted_y_array = np.ones(num_of_x)
        for i in range(num_of_x):
            predicted_y_array[i] = self.traverse_tree_and_get_value(points[i], 0)

        return predicted_y_array
