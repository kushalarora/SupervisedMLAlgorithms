from metrics import Metrics
from itertools import product
from datasets import Datasets
from constants import BINARY_CLASSIFIER

class Learn:
    def _check_type(self, metric):
        return self.type & 1 << metric

    def __init__(self, parameters={}, cross_validate=False, allowed_metrics=[], type_masks=[], test={}):
        type = 1
        self.parameters = parameters
        self.cross_validate = cross_validate
        self.allowed_metrics = allowed_metrics
        self.metrics = Metrics(type_masks)
        for mask in type_masks:
            type = type | mask
        self.type = type
        self.test = test

    def _cross_validate(self):
        """
        """
        params = self.parameters
        # Example {'a': 1, 'b': [1,2,3], 'c': [1,2])}

        self.parameters = {}
        lists = []
        for param, value in params.iteritems:
            if type(value) != list:
                # Values not to be modified
                # example param = 'a', value = 1
                lists.append([(param, value)])
            else:
                # Example = 'b' : [1,2,3]
                # output would be [('b',1), ('b',2), ('b',3)]]
                lists.append(product(param, value))

        # to pass list as *args
        # SO/3941517
        # Example
        # [
        #   (('a', 1),('b',1),('c', 1)), (('a', 1), ('b', 2), ('c',1)),
        #   (('a', 1), ('b', 3), ('c', 1)), (('a', 1), ('b',1),('c',2)),
        #   (('a', 1), ('b', 2), ('c', 2)), (('a', 1), ('b', 3), ('c',2))
        # ]

        tup_tuples = product(*lists)
        self._cross_validate_routine(tup_tuples)
        # TODO(Kushal): Incomplete. Think and finish.

    def _cross_validate_routine(self, tup_tuples):
        raise NotImplementedError

    def _train_routine(self, train_X, train_Y):
        raise NotImplementedError

    def train(self, train_X, train_Y):
        if self.cross_validate:
            self.parameters = self._cross_validate()
        return self._train_routine(train_X, train_Y)

    def predict(self, test_data=[]):
        raise NotImplementedError

    def evaluate(self, test_data, orig_Y, metrics=[]):
        results = []
        test_Y = self.predict(test_data)
        for metric in metrics:
            assert metric in self.allowed_metrics, "%s not a valid metric for %s" % (metric, self.__class__.__name__)
            metric_function = getattr(self.metrics, metric)
            results.append((metric, metric_function(orig_Y, test_Y)))
        return results

    def test_model(self):
        dt = Datasets()
        dataset = None
        if self._check_type(BINARY_CLASSIFIER):
            dataset = dt.load_iris()
        else:
            dataset = dt.load_digits()

        assert dataset

        self._train_routine(dataset[0], dataset[2])
        results = self.evaulate(dataset[1], dataset[3], self.test["metrics"])

        for tup in results:
            print "%s:" % tup[0]
            print tup[1]

