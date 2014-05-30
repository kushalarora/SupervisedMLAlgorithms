from prettytable import PrettyTable
import numpy as np

def print_accuracy_score(training_size, algorithm, dataset, score):
    table = PrettyTable(["training_size", "algorithm", "datset","metric", "score"])
    table.add_row([training_size, algorithm, dataset, "accuracy_score", score])
    print table

def print_precision_recall_fscore(algorithm, dataset, score):
    pass
    #print score).format('centered')

def print_cv_scores(algorithm, dataset, training_size, cv_results):
    table = PrettyTable(["parameters", "metric", "mean", "std"])
    print "Algorithm: %s, Dataset: %s, Training Size: %s" % (algorithm, dataset, training_size)
    for result in cv_results:
        table.add_row(result[:-1])
    print table


def print_breakdown(X, Y):
    print "ClassWise Breakdown"
    table = PrettyTable(["Class", "Size", "Percentage"])

    classes = np.unique(Y)

    class_count = [0 for c in classes]

    for i in xrange(0, len(X)):
        class_count[Y[i]] += 1

    for c in classes:
        table.add_row([c, class_count[c], float(class_count[c]) * 100/len(X)])

    print table
    print "\n################################################\n"



