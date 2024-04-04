# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:22:45 2024

@author: joels
"""


def calculate(new):
    
    #af = acceleration factor, default = 0.02, 0.02 = step. Maximum = 0.2
    ini_af = 0.02
    step_af = 0.02
    fin_af = 0.2
    
    new['trend'] = 0
    new['sar'] = 0.0
    new['real sar'] = 0.0               
    new['ep'] = 0.0                     #extreme price
    new['af'] = 0.0

    #Initialisation for recursive calculation
    new.iloc[1, new.columns.get_loc('trend')] = 1 if new['Close'].iloc[1] > new['Close'].iloc[0] else -1
    new.iloc[1, new.columns.get_loc('sar')] = new['High'].iloc[0] if new['trend'].iloc[1] > 0 else new['Low'].iloc[0]
    new.at[1, 'real sar'] = new['sar'].iloc[1]
    new.iloc[1, new.columns.get_loc('ep')] = new['High'].iloc[1] if new['trend'].iloc[1] > 0 else new['Low'].iloc[1]
    new.iloc[1, new.columns.get_loc('af')] = ini_af
    
    #Calculation
    for i in range(2, len(new)):
        
        temp = new['sar'].iloc[i-1] + new['af'].iloc[i-1] * (new['ep'].iloc[i-1] - new['sar'].iloc[i-1])
        if new['trend'].iloc[i-1] < 0:
            new.iloc[i, new.columns.get_loc('sar')] = max(temp, new['High'].iloc[i-1], new['High'].iloc[i-2])
            temp = 1 if new['sar'].iloc[i] < new['High'].iloc[i] else new['trend'].iloc[i-1] - 1
            
        else:
            new.iloc[i, new.columns.get_loc('sar')] = min(temp, new['Low'].iloc[i-1], new['Low'].iloc[i-2])
            temp = -1 if new['sar'].iloc[i] > new['Low'].iloc[i] else new['trend'].iloc[i-1] + 1
        new.iloc[i, new.columns.get_loc('trend')] = temp
    
        
        if new['trend'].iloc[i] < 0:
            temp = min(new['Low'].iloc[i], new['ep'].iloc[i-1]) if new['trend'].iloc[i] != -1 else new['Low'].iloc[i]
            
        else:
            temp = max(new['High'].iloc[i], new['ep'].iloc[i-1]) if new['trend'].iloc[i] != 1 else new['High'].iloc[i]
        new.iloc[i, new.columns.get_loc('ep')] = temp
    
        if np.abs(new['trend'].iloc[i]) == 1:
            temp = new['ep'].iloc[i-1]
            new.iloc[i, new.columns.get_loc('af')] = ini_af
        else:
            temp = new['sar'].iloc[i]
            if new['ep'].iloc[i] == new['ep'].iloc[i-1]:
                new.iloc[i, new.columns.get_loc('af')] = new['af'].iloc[i-1]
            else:
                new.iloc[i, new.columns.get_loc('af')] = min(fin_af, new['af'].iloc[i-1] + step_af)
        new.iloc[i, new.columns.get_loc('real sar')] = temp
       
    return new


def psar(new):
    
    calculate(new)
    new = new.drop(new.index[0])
    
    return new


#-------------------------------Example Usage---------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# path = #local dir
# df = pd.read_csv(path + f'data/{name}-USD.csv', index_col = 'Date', parse_dates=True)



# delta = df.tail(300)  #Sample

# delta = psar(delta)
# ax = delta[['Close','sar']].plot(color = ['black','red'])
# plt.show()
#------------------------------------------------------------------------------