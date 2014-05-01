import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab
import reductions
from matplotlib.colors import ListedColormap
from datasets import Datasets
from sklearn.preprocessing import scale

dts = Datasets(0)

def plot_data(filename, label, dataset, dimension=2, reduction='PCA'):
    data = dts.load_dataset(dataset, train_size=0)
    (X, Y) = (data['x_test'], data['y_test'])

    X = scale(X)
    # Reduce the dimensionality to the given value
    red_func = getattr(reductions, reduction)
    X = red_func(X, dimension)

    if dimension == 2:
        plot_2d(filename, label, X, Y)

def plot_2d(filename, label, X, Y, Z=None):
    assert X.shape[0] > 2, "Only two dimensional data allowed. Apply reduction to data first"
    h = 0.02
    figure = pylab.figure(figsize=(10, 10))
    cmap = pylab.cm.winter
    cmap.set_under("magenta")
    cmap.set_over("yellow")

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))

    pl = pylab.subplot(1, 1, 1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y, s=60, cmap=cmap)
    pl.set_xlim(xx.min(), xx.max())
    pl.set_ylim(yy.min(), yy.max())
    pl.set_xticks(np.linspace(xx.min(), 0.5, xx.max(), endpoint=True))
    pl.set_yticks(np.linspace(xx.min(), 0.5, xx.max(), endpoint=True))
    pl.set_title(label)
    figure.savefig(filename)

if __name__ == "__main__":
    plot_data("/tmp/test_plot", "Label:Iris", "iris")
