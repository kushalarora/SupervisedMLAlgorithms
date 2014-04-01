from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from learn import Learn

class KNeighborLearning(Learn):
    def __init__(self, **kwargs):
        Learn.__init__(self, kwargs)
        self.knn = KNeighborsClassifier()

    def _train_routine(self, train_X, train_Y):
        return self.knn.fit(train_X, train_Y)

    def test(self, test_data=[]):
        return self.knn.predict(test_data)


