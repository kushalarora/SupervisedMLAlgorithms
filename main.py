from datasets import Datasets
import argparse
import importlib
import logging
import yaml
import os
import sys
import importlib
import print_score
import shutil
from plot import plot_data
dt = Datasets()
def generate_pdf(metric_tuples):
    pass

def dump_results(metric_tuples):
    pass

def print_results(training_size, algorithm, dataset, metric_tuples):
    #print "\nFor Algorithm::\t%s" % algorithm
    #print "For Dataset::\t%s\n" % dataset
        for met_tup in metric_tuples:
            func = getattr(print_score, "print_%s" % met_tup[0])
            func(training_size, algorithm, dataset, met_tup[1])

def _get_algorithm_class(algorithm_name):
    module = importlib.import_module("%s" % algorithm_name)

    if not module:
        logging.error("Module %s not found" % algorithm_name)

    class_name = algorithm_name.replace("_"," ").title().replace(" ","")
    logging.info("Algorithm %s loaded from module %s" % (class_name, algorithm_name))
    return getattr(module, class_name)

def _load_dataset(dataset, dt):
    load_function = getattr(dt, "load_%s" % dataset)
    if not load_function:
        logging.error("Dataset %s couldn't be loaded" % dataset)
        sys.exit(0)

    return load_function()


def run_algorithms(algorithms, datasets, metrics, output, conf):
    dts = Datasets()
    if conf.get("plot_data", False):
        plot_dir = conf.get("plot_dir", "../plots")

        tmp_plot_dir = "/tmp/plots"
        if os.path.exists(tmp_plot_dir):
            shutil.rmtree(tmp_plot_dir)

        os.mkdir(tmp_plot_dir)

        for dataset in datasets:
            plot_data(os.path.join(tmp_plot_dir, "%s-orig.png"  % dataset), "%s-orig" % dataset, dataset)


    for algorithm in algorithms:

        algo_conf = conf["algorithms"].get(algorithm, None)

        if not algo_conf:
            logging.error("Algorithm %s not found in conf file" % algorithm)
            sys.exit(0)

        learn_class = _get_algorithm_class(algorithm)
        learn = learn_class(**algo_conf)
        learn._set_cross_validation(conf.get("cv_method", None), conf.get("cv_metric", None), conf.get("cv_params", None))
        results = []
        for training_size in conf.get("training_size", [0.40]):
            for dataset in datasets:

                if dataset not in conf["datasets"]:
                    logging.error("Dataset %s not found" % dataset)
                    sys.exit(0)

                data = dts.load_dataset(dataset, training_size)

                if learn.check_type(data["type"]):
                    eval_metrics = []
                    if metrics:
                        eval_metrics.extend(metrics)
                    else:
                        eval_metrics.extend(algo_conf["allowed_metrics"])

                    learn.train(data["x_train"], data["y_train"])
                    result_tups = learn.evaluate(data["x_test"], data["y_test"], eval_metrics)

                    if output == "print":
                        print_results(training_size, algorithm, dataset, result_tups)
                    else:
                        results.append((algorithm, dataset, result_tups))

                    if conf.get("plot_data", False):
                        output_path = os.path.join(tmp_plot_dir, "%s_%s_size_%d" % (dataset, algorithm, training_size))
                        output_label = "%s-%s-size-%s" % (dataset, algorithm, training_size)
                        learn.plot_results(output_path, output_label, data['x_train'], data['x_test'], data['y_train'], data['y_test'])
        if output == "pdf":
            generate_pdf(results)
        elif output == "dump_text":
            dump_results(results)
    if conf.get("plot_data", False):
        shutil.rmtree(plot_dir)
        shutil.move(tmp_plot_dir, plot_dir)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    FORMAT = '%(levelname)s %(asctime)s %(name)s: %(message)s'

    parser.add_argument("-c", "--config", type=str, default="./classification.json",
            help="Config file path")
    parser.add_argument("-a", "--algorithms", type=str, nargs='+',
            help="""Algorithm to run(ones mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """, default=['all'])


    parser.add_argument("-d", "--datasets", type=str, nargs='+',
            help="""Datasets to run(mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """, default=['all'])

    parser.add_argument("-m", "--metrics", type=str, nargs='+',
            help="""Metrics to evaluate(mentioned in classification.json):\n
                        values:: all allowed for a given algorithm(mentioned in json file) or specific ones\n
                """, default=['all'])

    parser.add_argument("-o", "--output", type=str,
            help="""Output Pattern values: [pdf(Generate a pdf report), print(Print to screen), dump(Dump in text file ./output.txt)
                """, default="print")

    logging.basicConfig(format=FORMAT, level=logging.INFO)
    args = parser.parse_args()

    logging.info("Main::Running algorithm(s): %s" % args.algorithms)
    logging.info("Main::Running on dataset(s): %s" % args.datasets)
    logging.info("Main::Evaluate metric(s): %s" % args.metrics)
    logging.info("Main::Output Mode: %s" % args.output)

    conf_file = open(os.path.abspath(args.config))
    conf_json = yaml.load(conf_file)

    algorithms = []
    datasets = []
    metrics = []

    if len(args.algorithms) and args.algorithms[0] == 'all':
        algorithms.extend(conf_json["algorithms"].keys())
    else:
        algorithms.extend(args.algorithms)

    if len(args.datasets) and args.datasets[0] == 'all':
        datasets.extend(conf_json["datasets"])
    else:
        datasets.extend(args.datasets)

    if len(args.metrics) > 1 and 'all' not in args.metrics:
        metrics.extend(args.metrics)
    run_algorithms(algorithms, datasets, metrics, args.output, conf_json)
