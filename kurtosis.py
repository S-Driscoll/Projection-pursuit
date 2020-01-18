from numpy import *

def kurtosis(x, flag=1, dim=None):
    # return k
    #KURTOSIS Kurtosis.
    #   K = KURTOSIS(X) returns the sample kurtosis of the values in X.  For a
    #   vector input, K is the fourth central moment of X, divided by fourth
    #   power of its standard deviation.  For a matrix input, K is a row vector
    #   containing the sample kurtosis of each column of X.  For N-D arrays,
    #   KURTOSIS operates along the first non-singleton dimension.
    #
    #   KURTOSIS(X,0) adjusts the kurtosis for bias.  KURTOSIS(X,1) is the same
    #   as KURTOSIS(X), and does not adjust for bias.
    #
    #   KURTOSIS(X,FLAG,'all') is the kurtosis of all the elements of X.
    #
    #   KURTOSIS(X,FLAG,DIM) takes the kurtosis along dimension DIM of X.
    #
    #   KURTOSIS(X,FLAG,VECDIM) finds the kurtosis of the elements of X based
    #   on the dimensions specified in the vector VECDIM.
    #
    #   KURTOSIS treats NaNs as missing values, and removes them.
    #
    #   See also MEAN, MOMENT, STD, VAR, SKEWNESS.
    
    #   Copyright 1993-2018 The MathWorks, Inc.

    # Validate flag
    if flag not in [0,1]:
        raise Exception('stats:trimmean:BadFlagReduction')

    if dim is None:
        if x.size == 0:
            # The output size for [] is a special case, handle it here.
            k = empty(x.shape)
            k[:] = nan
            return k
        else:
            # Figure out which dimension nanmean will work along.
            # First dimension with length > 1
            try:
                dim = [i for i,d in enumerate(x.shape) if d != 1][0]
            except IndexError: # all dimensions are length 1
                dim = 0
    
    # Center X, compute its fourth and second moments, and compute the
    # uncorrected kurtosis.
    x0 = x - nanmean(x,dim, keepdims=True)
    s2 = nanmean(x0**2,dim, keepdims=True) # this is the biased variance estimator
    m4 = nanmean(x0**4,dim, keepdims=True)
    k = m4 / s2**2
    
    # Bias correct the kurtosis.
    if flag == 0:
        n = sum(invert(isnan(x)).astype(int), dim, keepdims=True)
        n[n<4] = nan # bias correction is not defined for n < 4.
        k = ((n+1)*k - 3*(n-1)) * (n-1)/((n-2)*(n-3)) + 3
    return k
