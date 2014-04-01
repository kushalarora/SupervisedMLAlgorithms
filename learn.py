#from metrics import *
from itertools import product
class Learn:
    def __init__(self, parameters={}, cross_validate=False):
        self.parameters = parameters
        self.cross_validate = cross_validate

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

    def test(self, test_data=[]):
        raise NotImplementedError

    def evaulate(self, test_data=[], metrics=[]):
        test_Y = self.test(test_data)

