from metrics import Metrics
from itertools import product
from datasets import Datasets
from constants import BINARY_CLASSIFIER
from sklearn import cross_validation
from print_score import print_cv_scores
import logging


class Learn:
    def check_type(self, mask):
        return self.type & mask

    def __init__(self, parameters={}, cross_validate=False,
                    allowed_metrics=[], type_masks=[]):
        type = 1
        self.parameters = parameters
        self.cross_validate = cross_validate
        self.allowed_metrics = allowed_metrics
        self.metrics = Metrics(type_masks)
        self.algo = None
        for mask in type_masks:
            type = type | 1 << mask
        self.type = type
        self.cv = None

    def set_parameters(self, parameters={}):
        pass

    def _cross_validate(self, train_X, train_Y):
        """
        """
        params = self.parameters
        # Example {'a': 1, 'b': [1,2,3], 'c': [1,2])}

        lists = []
        for param, value in params.iteritems():
            if type(value) != list:
                # Values not to be modified
                # example param = 'a', value = 1
                lists.append([(param, value)])
            else:
                # Example = 'b' : [1,2,3]
                # output would be [('b',1), ('b',2), ('b',3)]]
                lists.append(product([param], value))

        # to pass list as *args
        # SO/3941517
        # Example
        # [
        #   (('a', 1),('b',1),('c', 1)), (('a', 1), ('b', 2), ('c',1)),
        #   (('a', 1), ('b', 3), ('c', 1)), (('a', 1), ('b',1),('c',2)),
        #   (('a', 1), ('b', 2), ('c', 2)), (('a', 1), ('b', 3), ('c',2))
        # ]

        tup_tuples = product(*lists)
        cv_results = []

        if not self.cross_validate:
            return dict(tup_tuples.next())

        for tup in tup_tuples:
            self.set_parameters(dict(tup))
            scores = cross_validation.cross_val_score(self.algo,
                                            train_X, train_Y,
                                            cv=self.cv_method(len(train_Y), *self.cv_params),
                                            scoring=self.cv_metric)
            cv_results.append((tup, self.cv_metric, scores.mean(), scores.std()))
        print_cv_scores(cv_results)
        max_tup = max(cv_results, key=lambda x:x[2])
        opt_parameters = dict(max_tup[0])
        logging.info("Choosing following parameters after validation %s" % opt_parameters)
        return opt_parameters


    def _set_cross_validation(self, method_name='KFold', metric='accuracy', parameters=[5]):
        if not self.cross_validate:
            return
        self.cv_params = parameters
        self.cv_metric = metric
        self.cv_method = getattr(cross_validation, method_name)

    def _train_routine(self, train_X, train_Y):
        raise NotImplementedError

    def train(self, train_X, train_Y):
        opt_parameters = self._cross_validate(train_X, train_Y)
        self.set_parameters(opt_parameters)
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

