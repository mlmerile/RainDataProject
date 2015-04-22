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
import scipy.integrate as integrate
import numpy as np

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



def hydrometeorType_dristribution(hydrometeorType) :
    return np.histogram( hydrometeorType,bins=range(16))[0] \
    / float(len(hydrometeorType))



def signalToHist(array,rang=(0,10),bins=5,density=False):
    return np.histogram( array,bins=bins,range=rang,density=density)[0]

def removeError(l) :
    return [ l[i] for i, elem in enumerate(l) if not (np.isnan(elem) or elem<-900\
     or elem>900) ]

def columnToHist(column):
    column = column.apply(removeError)
    hist=np.histogram([val for sublist in column.tolist() for val in sublist])
    mi=hist[1][0]
    ma=hist[1][-1]
    return column.apply(signalToHist,rang=(mi,ma),density=True) 


def dataToHist(data, columns,columnToHistFunction):
    
    res = data[columns].copy()    
    
    for j in columns : 
        res[j] = columnToHistFunction(res[j])

    return res
    
def dataFrameToMatrix(data) : 
    return np.column_stack( [ np.vstack(data[i]) for i in data.columns ] )
    
