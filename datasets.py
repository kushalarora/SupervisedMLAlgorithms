import numpy as np
from sklearn.cross_validation import train_test_split
from math import floor
from sklearn.datasets import load_iris, load_digits
from constants import *
import logging
import os
class Datasets:
    AVAILABLE_DATA = ['ocr_test', 'ocr_train', 'breast_cancer', 'higgs', 'iris', 'digits', 'london']
    DATA_FILES = {
            'ocr_test': 'optdigits.tes',
            'ocr_train': 'optdigits.tra',
            'breast_cancer': 'breast-cancer-wisconsin.data',
            'higgs': 'HIGGS.csv',
            'london_test': 'test.csv',
            'london_train': 'train.csv',
            'london_trainLabel': 'trainLabels.csv'
            }

    def __init__(self, training_size=0.40,cv_dir="../plot", dataset_dirs={
                            'ocr_test': '../OCR',
                            'ocr_train': '../OCR',
                            'breast_cancer': '../Wisconsin',
                            'higgs': '../Higgs',
                            'london_test': '../London',
                            'london_train': '../London',
                            'london_trainLabel': '../London'
                                    }):
        self.dataset_dirs = dataset_dirs
        self.train_size = training_size
        self.cv_dir = cv_dir

    def _load_file(self, dataset=None, path=None):
        assert dataset, "No dataset specified"
        file = None

        assert dataset in self.DATA_FILES, 'Dataset %s is not known' % dataset

        # Open file for reading
        try:
            file = open(os.path.join(self.dataset_dirs[dataset], self.DATA_FILES[dataset]))
        except IOError as e:
            logging.error('Dataset not found. Check dataset_dirs map for entry or specify one')
            raise e

        return file

    def _close_file(self, file):
        file.close()

    def load_dataset(self, dataset, train_size=None):
        assert dataset in self.AVAILABLE_DATA, "Dataset: %s not known" % dataset
        return getattr(self, "load_%s" % dataset)(train_size)

    def _build_output(self, type, X, Y, train_size=None, predict_on=None):
        train_size = train_size if train_size != None else self.train_size
        (x_train, x_test, y_train, y_test) = train_test_split(np.array(X),
                                                    np.array(Y),
                                                    train_size=train_size,
                                                    random_state=42)
        output = {
                'type': type,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'predict_on': predict_on,
                }
        return output


    def load_ocr_train(self, train_size=0.3):
        """ Load Optical Character Recoginition Test dataset
            MultiClass Dataset, UniLabel
            >>> dt = Datasets()
            >>> dt.load_ocr_train()
        """
        file = self._load_file('ocr_train')
        x_matrix = []
        y_vector = []
        for line in file:
            values = line.strip().split(',')
            y_vector.append(int(values[-1]))
            x_matrix.append(tuple([int(value) for value in values[:-1]]))

        self._close_file(file)
        return self._build_output(MULTICLASS_DATA, x_matrix, y_vector, train_size)

    def load_ocr_test(self, train_size=0.3):
        """ Load Optical Character Recoginition Train dataset
            >>> dt = Datasets()
            >>> dt.load_ocr_test()
        """
        file = self._load_file('ocr_test')
        x_matrix = []
        y_vector = []
        for line in file:
            values = line.strip().split(',')
            y_vector.append(int(values[-1]))
            x_matrix.append(tuple([int(value) for value in values[:-1]]))

        self._close_file(file)

        return self._build_output(MULTICLASS_DATA, x_matrix, y_vector, train_size)


    def load_breast_cancer(self, train_size=0.30):
        """ Load Winsconsin Breast Cancer dataset
            >>> dt = Datasets()
            >>> dt.load_breast_cancer()
        """
        file = self._load_file('breast_cancer')
        x_matrix = []
        y_vector = []
        for line in file:
            values = line.strip().split(',')
            y_vector.append(int(values[-1]))
            x_matrix.append(tuple([-1 if value == '?' else int(value) for value in values[1:-1]]))

        self._close_file(file)

        return self._build_output(BINARY_DATA,x_matrix, y_vector, train_size)

    def load_higgs(self, train_size=0.3, percentage=0.3):
        """ Load HIGGS dataset
            >>> dt = Datasets()
            >>> dt.load_higgs(percentage=0.3)
        """
        file = self._load_file('higgs')
        size = 0
        for line in file:
            size += 1
        self._close_file(file)

        indices = sorted(np.random.permutation(size)[:floor(size * percentage /100.0)])
        size = len(indices)
        file = self._load_file('higgs')
        count = 0
        idx = 0
        y_vector = []
        x_matrix = []
        for line in file:
            if size == idx:
                break

            if count == indices[idx]:
                values = line.strip().split(',')
                y_vector.append(int(float(values[0])))
                x_matrix.append(tuple([float(value) for value in values[1:]]))
                idx += 1
            count += 1

        self._close_file(file)
        return self._build_output(MULTICLASS_DATA, x_matrix, y_vector, train_size)

    def load_iris(self, train_size=0.3):
        x = load_iris()
        return self._build_output(MULTICLASS_DATA, x.data, x.target, train_size)


    def load_digits(self, train_size=0.3):
        x = load_digits()
        return self._build_output(MULTICLASS_DATA, x.data, x.target, train_size)

    def load_london(self, train_size=None):
        train_X_file = self._load_file('london_train')
        test_X_file = self._load_file('london_test')
        train_Y_file = self._load_file('london_trainLabel')

        train_X = []
        test_X = []
        train_Y = []
        for (X, file) in [(train_X, train_X_file), (test_X, test_X_file)]:
            for line in file:
               X.append(tuple([float(value) for value in line.strip().split(',')]))

        for line in train_Y_file:
            train_Y.append(int(line.strip()))

        self._close_file(train_X_file)
        self._close_file(train_Y_file)
        self._close_file(test_X_file)

        return self._build_output(BINARY_CLASSIFIER, train_X, train_Y, train_size, test_X)
