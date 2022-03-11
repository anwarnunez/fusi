import numpy as np
from scipy.stats import zscore


def cross_correlation(Y, X, zscorea=True, zscoreb=True):
    '''Compute correlation for each column of Y against
    every column of X (e.g. X is predictions).

    Parameters
    ----------
    Y n-by-p
    X n-by-q

    Returns
    -------
    corr (p-by-q) (true-by-pred)
    '''
    n = Y.shape[0]

    # If needed
    if zscorea:
        Y = zscore(Y)
    if zscoreb:
        X = zscore(X)
    corr = np.dot(Y.T, X)/float(n)
    return corr
