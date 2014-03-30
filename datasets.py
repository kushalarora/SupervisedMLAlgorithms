import numpy as np
from math import floor
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

    def load_ocr_train(self):
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
        return (np.array(x_matrix), np.array(y_vector))


    def load_ocr_test(self):
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
        return (np.array(x_matrix), np.array(y_vector))


    def load_breast_cancer(self):
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
        return (np.array(x_matrix), np.array(y_vector))

    def load_higgs(self, percentage=30):
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
        return (np.array(x_matrix), np.array(y_vector))


