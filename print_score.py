from prettytable import PrettyTable
def print_accuracy_score(algorithm, dataset, score):
    table = PrettyTable(["algorithm", "datset","metric", "score"])
    table.add_row([algorithm, dataset, "accuracy_score", score])
    print table

def print_precision_recall_fscore(algorithm, dataset, score):
    pass
    #print score).format('centered')

def print_cv_scores(cv_results):
    table = PrettyTable(["parameters", "metric", "mean", "std"])
    for result in cv_results:
        table.add_row(result)
    print table
