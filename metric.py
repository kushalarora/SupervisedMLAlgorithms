from sklearn import metrics

class Metric:
    BINARY_CLASSIFIER_METRIC = 1
    MULTICLASS_CLASSIFIER_METRIC = 2
    MULTILABEL_MULTICLASS_CLASSIFIER = 3
    def __init__(self, type):
        self.type = type

    def accuracy_score(self, y_orig, y_test, normalize=False):
        return metrics.accuracy_score(y_orig, y_test, normalize)

    def average_precision_score(self, y_orig, y_test):
        assert self.type != Metric.BINARY_CLASSIFIER_METRIC, "Average Precision Score works in binary classification case"
        return metrics.average_precision_score(y_orig, y_test)

    def confusion_matrix(self, y_orig, y_test):
        return metrics.confusion_matrix(y_orig, y_test)

    def
