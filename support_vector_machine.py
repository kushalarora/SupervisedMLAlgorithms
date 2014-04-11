from learn import Learn
import numpy
from sklearn.svm import NuSVC

class SupportVectorMachine(Learn):
    def __init__(self, **kwargs):
        Learn.__init__(self, **kwargs)
        self.svm = NuSVC();

    def _train_routine(self, train_X, train_Y):
        # define training routine for svm
        self.svm.fit(train_X, train_Y)


    def predict(self, test_X):
        # predict routine for svm
        return self.svm.predict(test_X)

