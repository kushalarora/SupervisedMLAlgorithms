from sklearn import metrics

class Metric:
    BINARY_CLASSIFIER_METRIC = 0
    MULTICLASS_CLASSIFIER_METRIC = 1
    MULTILABEL_CLASSIFIER = 2
    PROBABILISTIC_CLASSIFIER = 4

    def _check_type(self, metric):
        return self.type & metric

    def __init__(self, type_masks):
        type = 1
        for mask in type_masks:
            type | = mask
        self.type = type
        self.threshold = 0.5
        self.beta = 1.0

    def set_beta(self, beta):
        self.beta = beta
        return self

    def set_threshold(self, threshold):
        self.threshold = threshold
        return self

    def accuracy_score(self, y_orig, y_test, normalize=False):
        return metrics.accuracy_score(y_orig, y_test, normalize)

    def average_precision_score(self, y_orig, y_test):
        assert self._check_type(Metric.BINARY_CLASSIFIER_METRIC), "Average Precision Score works in binary classification case"
        return metrics.average_precision_score(y_orig, y_test)

    def confusion_matrix(self, y_orig, y_test):
        return metrics.confusion_matrix(y_orig, y_test)

    def zero_one_loss(self, y_orig, y_test):
        return metrics.zero_one_loss(y_orig, y_test)

    def matthews_corrcoef(self, y_orig, y_test):
        assert self._check_type(Metric.BINARY_CLASSIFIER_METRIC), "Matthews Correlation Coefficient is for binary classification"
        return metrics.matthews_corrcoef(y_orig, y_test)

    def log_loss(self, y_orig, y_test):
        assert self._check_type(Metric.PROBABILISTIC_CLASSIFIER), "Log Loss needs predicted values to be probabilities"
        return metrics.log_loss(y_orig, y_test)

    def precision_recall_fscore_average(self, y_orig, y_test):
        assert self.beta <= 1 and self.beta >= 0, "Beta is relative importance of recall vs precision. Must be between [0, 1]"
        return metrics.precision_recall_fscore_support(y_orig, y_test, beta=self.beta, average='weighted')

    def precision_recall_fscore(self, y_orig, y_test):
        assert self.beta <= 1 and self.beta >= 0, "Beta is relative importance of recall vs precision. Must be between [0, 1]"
        return metrics.precision_recall_fscore_support(y_orig, y_test, beta=self.beta, average=None)

