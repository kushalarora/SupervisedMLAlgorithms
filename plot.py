import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import pylab
import reductions
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from datasets import Datasets
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from constants import *
import os

def plot_histogram(X_data, y_data, label, bins=None, filename=None):

    dim = min(4, X_data.shape[1])

    X_red = reductions.PCA(X_data, dim)

    classes = np.unique(y_data)
    x_class = {}

    for c in classes:
        x_class[c] = []

    for i in xrange(0, len(y_data)):
        x_class[y_data[i]].append(tuple(X_red[i]))

    for c in classes:
        x_class[c] = np.array(x_class[c])

    fig = plt.figure()

    for i in xrange(0, dim):
        data = []
        data.append(X_red[:, i])
        for c in classes:
            data.append(tuple(x_class[c][:, i]))

        ax = pylab.subplot(dim, 1, i + 1)
        ax.set_title("%s-%d" % (label, i))
        ax.hist(data, histtype='bar', fill=True)
        ax.legend()
    if filename:
        fig.savefig(filename)
    else:
        fig.show()

def plot_scatter(X, Y, label, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = pylab.cm.winter
    cmap.set_under("magenta")
    cmap.set_over("yellow")
    X = reductions.PCA(X, 3)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    z_min, z_maz = X[:, 2].min() - 5, X[:, 2].max() + .5

    # Plot the training points
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(x_min, x_max, 10))
    ax.set_yticks(np.linspace(y_min, y_max, 10))
    ax.set_title(label)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, s=60, cmap=cmap)
    ax.legend()
    if filename:
        fig.savefig(filename)
    else:
        fig.show()


def plot_confusion_matrix(y_test, y_pred, label, filename=None):
    cm = confusion_matrix(y_test, y_pred)

    pylab.subplot(111)
    pylab.matshow(cm)
    # Show confusion matrix in a separate window
    pylab.title(label, fontsize=20)
    pylab.colorbar()
    pylab.ylabel('True label', fontsize=20)
    pylab.xlabel('Predicted label', fontsize=20)
    if filename:
        pylab.savefig(filename)
    else:
        pylab.figshow()

def plot_metric(type, y_test, y_pred, dataset, algorithm, training_size, filename=None):
    label = "confusion_matrix-%s-%s-size-%d" % (dataset, algorithm, training_size)
    plot_confusion_matrix(y_test, y_pred, label, filename)


def plot_cv(algorithm, dataset, training_size, cv_results, cv_dir):
    label = "cross_validation-%s-%s-size-%s" % (algorithm, dataset, training_size)
    filename = os.path.join(cv_dir, "cv-%s-%s-size-%s.png" % (algorithm, dataset, training_size))
    return
    n_dim = len(cv_results[0][0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    pylab.show()


def plot_training_results(train_sizes, scores, filename=None):
    width = 7
    fig = pylab.figure()
    ax = pylab.subplot(111)
    ax.bar(train_sizes, scores, width, color='b')
    ax.set_ylabel('Scores')
    ax.set_xticks(train_sizes)
    ax.set_title('Accuracy vs Training Size')
    fig.show()
    fig.savefig(filename)


def plot_PCA_variance(f_var_ratio, label, filename=None):
    fig = pylab.figure()
    ax = pylab.subplot(111)
    ax.plot(np.vstack((np.array([1]) + f_var_ratio)))
    ax.set_ylabel('variance loss')
    ax.set_xlabel('# of features retained')
    ax.set_xticks(xrange(0, len(f_var_ratio) + 1))
    ax.set_yticks(xrange(0, 100))
    if filename:
        fig.savefig(filename)
    else:
        fig.show()

if __name__ == "__main__":
    plot_data("/tmp/test_plot", "Label:Iris", "iris")
