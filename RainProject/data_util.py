"""
The Data Module
===============

This module give functionalities to manipulate Datas.

The main representation of the Data is the Dataframe of Panda Library.
"""
import pandas as pd
import scipy.interpolate as sci
import sys
import logging
import scipy.integrate as integrate
import numpy as np

## DEBUG
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def cut(l):
    """
    Extract the sequences that count down to zero

    :param l: list to cut
    """
    last_jump = 0
    for i in xrange(1,len(l)):
        if l[i-1] <= l[i]:
            yield l[last_jump:i]
            last_jump = i
    yield l[last_jump:i+1]

def split_raindata(s):
    """
    Specific split function to handle the fact that some element are
    string and others are numeric

    :param s: cell of a column
    """
    if isinstance(s,(int,long,float)):
        return [s]
    else:
        return s.split()

def col_str_to_list(col,unchanged_cols):
    if col.name in unchanged_cols:
        return col
    else:
        return col.map(lambda x: map(float,split_raindata(x)))
    
def read_raincsv(filename,nrows=None,unchanged_cols=("Id","Expected")):
    """
    Read a csv and return a well-typed dataframe

    :param filename: The name of csv file
    :param unchanged_cols: cols well typed already
    """

    logging.debug("Start reading file")
    
    data = pd.read_csv(filename,nrows=nrows)

    logging.debug("End reading file")

    data = data.apply(col_str_to_list,axis=0,args=(unchanged_cols,))

    logging.debug("End of the processing (reading file)")
    return data

def drop_data(d):
    """
    Drop useless columns

    :param d: dataframe
    """
    return d.drop("Kdp")

def select_not_nan(x,y):
    return ([x[i] for i, elem in enumerate(y) if not pd.isnull(elem)],
            [y[i] for i, elem in enumerate(y) if not pd.isnull(elem)])

def extrapolate_cell(x,xi,yi,spline_order=3):
    s = sci.InterpolatedUnivariateSpline(np.fliplr([xi])[0],np.fliplr([yi])[0],k=spline_order)
    return np.fliplr([s(np.fliplr([x])[0])])[0]


def select_not_error(t,x) : 
    return (np.array([t[i] for i, elem in enumerate(x) if elem>=0.0]),
            np.array([x[i] for i, elem in enumerate(x) if elem>=0.0]))

def reducetomean_signal(t,x):
    """

    :param t : list or np array , x = list or np array
    """
    #remove error and nan
    (t,x) = select_not_nan(t,x)
    (t,x) = select_not_error(t,x)
    if len(x) == 0 :
        return float("NaN")
    if len(x) == 1 :
        return x[0]

    return (integrate.trapz(x,t) / ( t[-1] - t[0] ))


    
def reducetomean(x):
    return np.mean( x )

def DistanceToRadarToIndex(dist):
    res=list()
    left=dist.copy()
    while not len(left) == 0 : 
        res.append(dist==left[0])
        left=left[np.logical_not(left==left[0])]
    return res

def averageOverRadar(f,indices,t,x):
    """

    :param f : function, indices : list of np array, (t,x) = (np array ,np array)
    """
    res = list()
    for l in indices : 
        res.append( f(t[l],x[l]) )
    return np.mean([ res[i] for i, elem in enumerate(res) if not np.isnan(elem) ] )


def reduce_data(data, signalFunction):
    
    res=data.copy()
    nbSample = len(data)
    
    for i in range(nbSample) : 
        
        #Distance to radar
        index = DistanceToRadarToIndex(np.array(data.DistanceToRadar[i]))
        res.DistanceToRadar[i] = reducetomean(data.DistanceToRadar[i])
                
        t = np.array(data.TimeToEnd[i])
        for j in range(3,19):
            res.iloc[i,j] = averageOverRadar(signalFunction,\
            index, t, np.array(data.iloc[i,j]))


    return res


