"""
Histogram utilities
===============

This module give functionalities to histogram
"""

import data_util as du
from pyemd import emd
import numpy as np
from scipy import stats
from collections import Counter
from functools import partial


def hydrometeorType_dristribution(hydrometeorType) :
    return np.histogram( hydrometeorType,bins=range(16))[0] \
    / float(len(hydrometeorType))

def signalToHist(array,rang=(0,10),bins=5,density=False):
    return np.histogram( array,bins=bins,range=rang,density=density)[0]


def columnToHist(column):
    column = column.apply(du.removeError)
    hist=np.histogram([val for sublist in column.tolist() for val in sublist])
    mi=hist[1][0]
    ma=hist[1][-1]
    return column.apply(signalToHist,rang=(mi,ma),density=True) 

def signal_to_quantilehist(s,bin_edges):
    hist,bins = np.histogram(s,bins=bin_edges,density=False)
    if np.sum(hist) != 0:
        hist = hist/float(np.sum(hist))
    return hist

def bin_quantilehist(unpacked_col,nb_bins):
    bin_edges = stats.mstats.mquantiles(unpacked_col,np.arange(nb_bins+1)/float(nb_bins))
    dict_bin_edges = Counter(bin_edges)
    bin_edges = sorted(sum([min(2,val)*(key,) for key,val in list(dict_bin_edges.items())],()))
    for i,pair in enumerate(du.pairwise(bin_edges)):
        a,b = pair
        if a == b:
            bin_edges[i+1] += float(1.0e-10)
     
    return bin_edges

def column_to_quantilehist(column,nb_bins):
    column = column.apply(du.removeError)
    unpacked_col = [val for sublist in column.tolist() for val in sublist]
    bin_edges = bin_quantilehist(unpacked_col,nb_bins)
    return column.apply(signal_to_quantilehist,args=(bin_edges,))

def build_col_to_quantilehist(nb_bins):
    return partial(column_to_quantilehist,nb_bins=nb_bins)

def calc_bin_quantile(data,nb_bins):
    res = []
    for name,col in data.iteritems():
        if col.dtype == np.dtype(object):
            unpacked_col = [val for sublist in col.tolist() for val in sublist]
            bin_edges = bin_quantilehist(unpacked_col,nb_bins)
            res.append(bin_edges)

    return res

def dataToHist(data, columns,columnToHistFunction):
    res = data[columns].copy()    
    
    for j in columns : 
        res[j] = columnToHistFunction(res[j])

    return res
    

    
