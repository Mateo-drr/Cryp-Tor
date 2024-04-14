# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:20:21 2024

@author: Mateo-drr
"""

import numpy as np
import pandas as pd

def getHalving(df):
    dates = ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19']
    
    # Create a datetime index using pd.date_range()
    date_index = pd.date_range(start=dates[0], end=dates[-1])
    # Create a DataFrame with the datetime index
    alldates = pd.DataFrame(index=date_index)
    
    #find indexes of the dates in the df
    idx=[]
    daysleft=[]
    for day in dates:
        pos = alldates.index.get_loc(day)
        idx.append(pos)
    
    daysleft.append(np.linspace(0, 100, idx[1]-idx[0]))
    daysleft.append(np.linspace(0, 100, idx[2]-idx[1]))
    daysleft.append(np.linspace(0, 100, idx[3]-idx[2]))
    daysleft.append(np.linspace(0, 100, idx[4]-idx[3]))
    daysleft = np.concatenate(daysleft,axis=0)
    daysleft = np.hstack((100,daysleft))
    
    alldates['halv'] = daysleft
    alldates.index = pd.to_datetime(alldates.index)
    
    # Get the start and end dates of the 'df' DataFrame
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Slice the 'alldates' DataFrame based on the start and end dates
    sliced_alldates = alldates[start_date:end_date]
    
    #join with the coin df
    df.loc[:,'halv'] = sliced_alldates.values
    
    return df


'''
date_to_find = pd.to_datetime('2024-03-25')  # The date you want to find

# Boolean indexing to select rows where the date matches
row = df[df.index == date_to_find]

print(row)

'''