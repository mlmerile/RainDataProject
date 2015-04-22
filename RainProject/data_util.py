"""
The Data Module
===============

This module give functionalities to manipulate Datas.

The main representation of the Data is the Dataframe of Panda Library.
"""
import pandas as pd
import scipy.interpolate as sci
import itertools as it
import numpy as np
import sys
import logging

## DEBUG
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)   

def chunks(l,c):
    """
    Cut the list at the position in c
    """
    return [l[i:j] for i,j in pairwise([0] + c)]

def cut(l):
    """
    Extract the sequences that count down to zero

    :param l: list to cut
    """
    last_jump = 0
    i = 0
    for i in xrange(1,len(l)):
        if l[i-1] <= l[i]:
            yield l[last_jump:i]
            last_jump = i
    yield l[last_jump:i+1]

def separate_inter(x):
    c = np.cumsum(map(len,list(cut(x["TimeToEnd"])))).tolist()
    res = []
    series = []
    if len(c) >= 2:
        for i,elem in x.iteritems():
            if isinstance(elem,list):
                chunks_elem = chunks(elem,c)
            else:
                chunks_elem = [elem]*len(c)
            res.append(chunks_elem)

        for i in range(len(c)):
            series.append(pd.Series([elemj[i] for elemj in res],index=x.index))
            
        df = pd.concat(series,axis=1).T
        return df

def separate_radar(d):
    df = pd.concat([separate_inter(row) for i,row in d.iterrows()],axis=0)
    print df
    return pd.concat([d,df], axis=0)

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
    return d.drop("Kdp",1)

def error_code_to_nan(x):
    if x == -99900 or x == -99901 or x == -99903 or x == 999.0:
        return np.nan
    else:
        return x

def set_to_nan(d):
    """
    Set error codes to nan
    """
    return d.applymap(lambda x: map(error_code_to_nan,x) if isinstance(x,list) else x)

def select_not_nan(x,y):
    return ([x[i] for i, elem in enumerate(y) if not pd.isnull(elem)],
            [y[i] for i, elem in enumerate(y) if not pd.isnull(elem)])

def extrapolate_cell(x,xi,yi,spline_order):
    order = min(len(xi)-1,spline_order)
    if order < 1:
        if order < 0:
            s = np.nan
        else:
            s = [yi[0] for i in range(len(x))]
    else:
        s = sci.InterpolatedUnivariateSpline(np.fliplr([xi])[0],np.fliplr([yi])[0],k=order)
        s = np.fliplr([s(np.fliplr([x])[0])])[0]
    return s

def extrapolate_inter(s,x_extra,spline_order):
    final_x = x_extra
    res = []

    res.append(s["Id"])
    current_x = s["TimeToEnd"];
    res.append(final_x)
    
    for i,elem in s.iteritems():
        if i != "TimeToEnd" and i != "Expected" and i != "Id":
            current_y = elem
            cleaned_current_x,cleaned_current_y = select_not_nan(current_x,current_y)
            final_y = extrapolate_cell(final_x,cleaned_current_x,cleaned_current_y,spline_order)
            res.append(final_y)

    res.append(s["Expected"])
    return pd.Series(res,index=s.index)
    
def extrapolate(d,x_extra=np.arange(61,-1,-1),spline_order=3):    
    return d.apply(extrapolate_inter,axis=1,args=(x_extra,spline_order))

def clean_data(data,x_extra=np.arange(61,-1,-1),spline_order=3):
    data = drop_data(data)
    data = separate_radar(data)
    data = set_to_nan(data)
    data = extrapolate(data,x_extra,spline_order)

    return data


