"""
Continuous Ranked Probability Score
===================================

Evaluate the prediction using the continuous ranked probability score.

A simple matrix representation is used in this case.
"""
import numpy as np
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def is_cdf(l):
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1)) \
      and l[0] >= 0 and l[-1] <= 1

def heavyside(truth):
    return np.array([(np.arange(70) >= i) for i in truth])

def crps(predictions,truth):
    worst_case = 1.0
    mat_is_cdf = np.apply_along_axis(is_cdf,1,predictions)
    if np.all(mat_is_cdf):
        return np.mean(np.square(predictions-heavyside(truth)))
    else:
        logging.debug("Non CDF prediction")
        return worst_case 

