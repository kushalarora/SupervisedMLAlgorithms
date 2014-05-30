from datasets import Datasets
import importlib
import os
import sys
import importlib
import logging
import print_score
import shutil
import yaml
import constants
from plot import plot_scatter, plot_histogram, plot_metric, plot_PCA_variance
from sklearn.decomposition import PCA
dt = Datasets()

class ClassifierLib:
    def __init__(self, analyze, conf, data_conf, output, algorithms, datasets, metrics):
        self.conf = self._load_conf(conf)
        self.shall_analyze = analyze
        self.data_conf = self._load_conf(data_conf)
        self.data_class = Datasets()
        self.output = output
        self.output_dir = os.path.abspath(self.conf.get("output_dir", "./output"))
        self.shall_plot = self.conf.get("plot_data")
        self.algorithms = algorithms
        self.datasets = datasets
        self.metrics = metrics


    def _get_algorithm_class(self, algorithm_name):
        module = importlib.import_module("%s" % algorithm_name)

        if not module:
            logging.error("Module %s not found" % algorithm_name)

        class_name = algorithm_name.replace("_"," ").title().replace(" ","")
        logging.info("Algorithm %s loaded from module %s" % (class_name, algorithm_name))
        return getattr(module, class_name)

    def _load_conf(self, conf_path):
        conf_file = open(os.path.abspath(conf_path))
        return yaml.load(conf_file)


    def run_algorithm(self, algorithm, data, data_conf, training_size):
        algo_conf = self.conf['algorithms'][algorithm]
        learn_class = self._get_algorithm_class(algorithm)
        learn = learn_class(**algo_conf)

        if not learn.check_type(getattr(constants, data_conf["type"])):
            return

        dataset = data_conf['name']

        learn.set_dataset(dataset, training_size)

        if algo_conf.get("cross_validate", False):
            learn._set_cross_validation(self.conf.get("cv_method", None), self.conf.get("cv_metric", None), self.conf.get("cv_params", None))
            learn.cross_validation(data['x_train'], data['y_train'], self.conf.get('print_cv_score', self.conf.get('print_cv_score', False)))

        learn.train(data["x_train"], data["y_train"])
        result = learn.predict(data['x_test'])

        if self.conf.get('evaluate', False):
            eval_metrics = []
            if self.metrics:
                eval_metrics.extend(self.metrics)
            else:
                eval_metrics.extend(algo_conf["allowed_metrics"])

            result = learn.evaluate(result, data["y_test"], eval_metrics)

        return result

    def run(self):
        if os.path.exists(self.output_dir):

            if os.path.exists("%s%s" % (self.output_dir, "_1")):
                shutil.rmtree("%s%s" % (self.output_dir, "_1"))

            shutil.move(self.output_dir, "%s%s" % (self.output_dir, "_1"))

        os.mkdir(self.output_dir)

        for dataset in self.datasets:
            if dataset not in self.data_conf:
                logging.error("Dataset %s not found" % dataset)
                sys.exit(0)

            dataset_dir = os.path.join(self.output_dir, dataset)
            os.mkdir(dataset_dir)

            if self.shall_analyze:
                self.analyze(dataset, dataset_dir)

            for algorithm in self.algorithms:

                algo_dir = os.path.join(dataset_dir, algorithm)
                os.mkdir(algo_dir)


                results = []
                for training_size in self.conf.get('training_sizes', [.4]):

                    data_conf = self.data_conf[dataset]

                    data = self.data_class.load_dataset(dataset, training_size)

                    result = self.run_algorithm(algorithm, data, data_conf, training_size)

                    if self.conf.get('evaluate', True):
                        if self.output == "print":
                            self.print_results(training_size, algorithm, dataset, result)

                        if self.shall_plot:
                            for metric, y_test, score in result:
                                metric_plot_path = os.path.join(algo_dir, "metric-%s-%s_%s_size_%d.png" % (metric, dataset, algorithm, training_size * 100))
                                plot_metric(data['type'], y_test, data['y_test'], dataset, algorithm, training_size * 100, metric_plot_path)
                    else:
                        result_file = open(os.path.join(algo_dir, "result.csv"), 'a+')
                        result_file.write(",".join(results))
                        result_file.close()


    def analyze(self, dataset, dataset_dir):
        data = self.data_class.load_dataset(dataset, train_size=100)

        (X, Y) = (data['x_train'], data['y_train'])
        print_score.print_breakdown(X, Y)

        if self.shall_plot:
            plot_scatter(X, Y, "%s-orig" % dataset, filename=os.path.join(dataset_dir, "%s-orig.png"  % dataset))
            plot_histogram(X, Y, "%s-hist" % dataset, filename=os.path.join(dataset_dir, "%s-hist.png" % dataset))

            pca = PCA()
            pca.fit(X)
            plot_PCA_variance(pca.explained_variance_ratio_ * 100, "%s-pca-#feature-vs-variance" % dataset, filename=os.path.join(dataset_dir, "%s-pca-variance-ratio" % dataset))


    def print_results(self, training_size, algorithm, dataset, metric_tuples):
        #print "\nFor Algorithm::\t%s" % algorithm
        #print "For Dataset::\t%s\n" % dataset
            for met_tup in metric_tuples:
                func = getattr(print_score, "print_%s" % met_tup[0])
                func(training_size, algorithm, dataset, met_tup[2])

