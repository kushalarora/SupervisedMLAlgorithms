from sklearn import decomposition
from sklearn import manifold

def PCA(X, dimensions):
    return decomposition.PCA(n_components=dimensions ).fit_transform(X)

def LinearEmbedding(X, dimension):
    return manifold.LocallyLinearEmbedding(3, dimension, method='standard').fit_transform(X)


