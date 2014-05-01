from metrics import Metrics
from itertools import product
from datasets import Datasets
from constants import *
from sklearn import cross_validation
from print_score import print_cv_scores
from reductions import PCA, LinearEmbedding
import logging
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split


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

    def _cross_validate(self, train_X, train_Y, print_scores=True):
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
        if print_scores:
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

    def train(self, train_X, train_Y, print_cv_score=True):
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

    def plot_results(self, filename, label, X_train, X_test, y_train, y_test):
        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((y_train, y_test))
        training_size = X_train.shape[0] * 100/X.shape[0]
        X = LinearEmbedding(X, 2)
        (X_train, X_test, y_train, y_test) = train_test_split(X, Y, train_size=training_size, random_state=42)

        self.train(X_train, y_train)
        score = self.algo.score(X_test, y_test)


        assert X.shape[0] > 2, "Only two dimensional data allowed. Apply reduction to data first"
        h = 0.02
        figure = pl.figure(figsize=(10, 10))
        ax = pl.subplot(1, 1, 1)
        cm = pl.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                      np.arange(y_min, y_max, h))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if self.check_type(ONE_VS_ALL):
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(self.algo, "decision_function"):
            Z = self.algo.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(self.algo, "predict_prob"):
            Z = self.algo.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            return

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)


        # Plot also the training points
        #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, s=30)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=50,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(label)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                  size=15, horizontalalignment='right')
        figure.savefig(filename)
