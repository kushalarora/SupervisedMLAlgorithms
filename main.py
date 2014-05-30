import argparse
import logging
import os
import yaml
from classifier import ClassifierLib

ALGO_CONF_PATH="./algorithms.json"
DATA_CONF_PATH="./dataset.json"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    FORMAT = '%(levelname)s %(asctime)s %(name)s: %(message)s'

    parser.add_argument("-c", "--config", type=str, default="./classification.json",
            help="Config file path")


    parser.add_argument("--dataset_conf", type=str, default="./datasets.json",
            help="Dataset config file path")

    parser.add_argument("--analyze", type=bool, default=True,
            help="Analyze data")

    parser.add_argument( "--algo_conf", type=str, default="./algorithms.json",
            help="Algorithms config file path")

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
        algorithms.extend(conf_json["algorithms"])
    else:
        algorithms.extend(args.algorithms)


    if len(args.datasets) and args.datasets[0] == 'all':
        datasets.extend(conf_json["datasets"])
    else:
        datasets.extend(args.datasets)

    if len(args.metrics) > 1 and 'all' not in args.metrics:
        metrics.extend(args.metrics)

    clf = ClassifierLib(
            args.analyze,
            args.config,
            args.dataset_conf,
            args.output,
            algorithms,
            datasets,
            metrics)

    clf.run()
