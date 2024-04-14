# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:01:10 2024

@author: joels
"""

def getBBAND(df,period,m):
    
    avg_hlc = (df.High + df.Low + df.Close)/3
    mean = avg_hlc.rolling(period).mean()
    std = avg_hlc.rolling(period).std()
    
    upband = mean + std*m
    midband = mean
    lowband = mean - std*m
    
    upband = upband.fillna(0)
    midband = midband.fillna(0)
    lowband = lowband.fillna(0)
    
    return upband, midband, lowband



#Note: m = standard deviation multiplier which is usually 2
#      period = smoothening period usually 20


#----------------------------Sample Usage--------------------------------------
# import matplotlib.pyplot as plt
# import pandas as pd


# path = #local dir
# df = pd.read_csv(path + f'data/{name}-USD.csv', index_col = 'Date', parse_dates=True)

# df["BBUp"],df["BBMid"],df["BBLow"] = getBBAND(df,20,2)

# delta = df.tail(200)  #Sample

# ax = delta[['Close', 'BBUp','BBMid','BBLow']].plot(color = ['black','green','gray','red'])
# ax.fill_between(delta.index, delta['BBUp'], delta['BBLow'], facecolor='orange', alpha=0.1)
# plt.show()
#------------------------------------------------------------------------------