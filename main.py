from datasets import Datasets
import argparse
import importlib
import json
import logging
import os
import sys

def generate_pdf(metric_tuples):
    pass

def dump_results(metric_tuples):
    pass

def print_results(algorithm, metric_tuples):
    print "Algorithm => %s:" % algorithm
    for (metric, result) in metric_tuples:
        print "\tMetric => %s:" % metric
        print "\t\t%s" % result


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
    for algorithm in algorithms:
        algo_conf = conf["algorithms"].get(algorithm, None)

        if not algo_conf:
            logging.error("Algorithm %s not found in conf file" % algorithm)
            sys.exit(0)

        learn_class = _get_algorithm_class(algorithm)
        learn = learn_class(**algo_conf)

        results = []
        dt = Datasets()
        for dataset in datasets:
            if dataset not in conf["datasets"]:
                logging.error("Dataset %s not found" % dataset)
                sys.exit(0)


            (x_train, x_test, y_train, y_test) = _load_dataset(dataset, dt)

            if not metrics:
                metrics = algo_conf["allowed_metrics"]

            learn.train(x_train, y_train)
            results = []
            metric_tups = learn.evaluate(x_test, y_test, metrics)

            if output == "print":
                print_results(algorithm, metric_tups)
            else:
                results.append((algorithm, metric_tups))

    if output == "pdf":
        generate_pdf(results)
    elif output == "dump_text":
        dump_results(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    FORMAT = '%(levelname)s %(asctime)s %(name)s: %(message)s'

    parser.add_argument("-c", "--config", type=str, default="./classification.json",
            help="Config file path")
    parser.add_argument("-a", "--algorithms", type=str, nargs='+',
            help="""Algorithm to run(ones mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """)


    parser.add_argument("-d", "--datasets", type=str, nargs='+',
            help="""Datasets to run(mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """)

    parser.add_argument("-m", "--metrics", type=str, nargs='+',
            help="""Metrics to evaluate(mentioned in classification.json):\n
                        values:: all allowed for a given algorithm(mentioned in json file) or specific ones\n
                """)

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
    conf_json = json.load(conf_file)

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
