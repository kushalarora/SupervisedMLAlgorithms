import numpy as np
from sklearn.cross_validation import train_test_split
from math import floor
from sklearn.datasets import load_iris, load_digits
import os
class Datasets:
    DATA_FILES = {
            'ocr_test': 'optdigits.tes',
            'ocr_train': 'optdigits.tra',
            'breast_cancer': 'breast-cancer-wisconsin.data',
            'higgs': 'HIGGS.csv'
            }

    def __init__(self, dataset_dirs={
                            'ocr_test': '../OCR',
                            'ocr_train': '../OCR',
                            'breast_cancer': '../Wisconsin',
                            'higgs': '../Higgs'
                                    }):
        self.dataset_dirs = dataset_dirs

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

    def load_dataset(self, dataset, train_size):
        assert dataset in self.dataset_dirs, "Dataset: %s not known" % dataset
        return getattr(self, "load_%s" % dataset)(train_size)

    def load_ocr_train(self, train_size=30):
        """ Load Optical Character Recoginition Test dataset
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
        return train_test_split(np.array(x_matrix), np.array(y_vector), train_size=train_size, random_state=42)


    def load_ocr_test(self, train_size=30):
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
        return train_test_split(np.array(x_matrix), np.array(y_vector), train_size=train_size, random_state=42)


    def load_breast_cancer(self, train_size=30):
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
        return train_test_split(np.array(x_matrix), np.array(y_vector), train_size=train_size, random_state=42)

    def load_higgs(self, train_size=30, percentage_data=30):
        """ Load HIGGS dataset
            >>> dt = Datasets()
            >>> dt.load_higgs(percentage=30)
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
        return train_test_split(np.array(x_matrix), np.array(y_vector), train_size=train_size, random_state=42)

    def load_iris(self, train_size=30):
        x = load_iris()
        return train_test_split(x.data, x.target, train_size=train_size, random_state=42)

    def load_digits(self, train_size=30):
        x = load_digits()
        return train_test_split(x.data, x.target, train_size=train_size, random_state=42)
