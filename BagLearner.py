import numpy as np
class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learners = [learner(**kwargs) for l in range(0, bags)]
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.node = None
        self.tree = None
        pass

    def author(self):
        return "ashabou3"

    def add_evidence(self, train_x, train_y):
        if self.verbose:
            pass
        num_of_data_points = train_x.shape[0]
        num_of_features = train_x.shape[1]
        for learner in self.learners:
            ran_int_arr = np.random.randint(0, num_of_data_points, size=(num_of_data_points))
            bag_train_x = np.random.random(size=(num_of_data_points, num_of_features))
            bag_train_y = np.random.random(size=(num_of_data_points))
            for x, d in enumerate(bag_train_x):
                r = ran_int_arr[x]
                bag_train_x[x] = train_x[r]
                bag_train_y[x] = train_y[r]
            learner.add_evidence(bag_train_x, bag_train_y)  # train it

    def query(self, points):
        predicted_y_array = [learner.query(points) for learner in self.learners]
        predicted_y = np.sum(predicted_y_array, axis=0)
        predicted_y[predicted_y < -1] = -1
        predicted_y[predicted_y > 1] = 1
        return predicted_y

