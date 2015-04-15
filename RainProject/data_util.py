"""
The Data Module
===============

This module give functionalities to manipulate Datas.

The main representation of the Data is the Dataframe of Panda Library.
"""
import pandas as pd
import sys
import logging

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

