# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:38:06 2024

@author: Mateo-drr
"""
import pandas as pd
import matplotlib.pyplot as plt
from indicators.psar import getPSAR
from indicators.bollingerband import getBBAND
from indicators.halving import getHalving
from indicators.rsi import getRSI
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import copy
import torch

coins=['BTC',
       'ETH',
       'XRP',
       'BNB',
       'DOGE',
       'ADA',
       'SOL',
       'DOT',
       ]
path = 'S:/Cryp-Tor/'
dframes = []

def readData(name):
    df = pd.read_csv(path + f'data/{name}-USD.csv', index_col = 'Date', parse_dates=True)
    df.drop(columns=['Adj Close'], inplace=True)
    df.head(5)
    return df

def plotData(df,name):
    plt.plot(df.Close)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.savefig(path + f"plots/{name}_initial_plot.png", dpi=250)
    plt.show();

# split a multivariate sequence past, future samples (X and y)
# TLDR it organizes the data, so that for example you have to predict
# 3 days given 10 past days
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    n_steps_out+=1
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def formatData(df,trainWindow,predWindow,split=True):
    X, y = df, df
    X.shape, y.shape
    
    #Re-scale the data 
    mm = MinMaxScaler(feature_range=(0,1)) #limit betwee 0 and 1
    ss = StandardScaler() #mean 0 and std 1
    X_trans = ss.fit_transform(X)
    y_trans = ss.transform(y)#.reshape(-1,1)#mm.fit_transform(y.reshape(-1, 1))
    
    if split:
        # Remove test and validation data
        total_samples = len(X_trans)
        train_test_cutoff = round(0.12 * total_samples)
        
        # Define indices for test, validation, and training data
        test_start_index = total_samples - train_test_cutoff
        val_start_index = test_start_index - train_test_cutoff * 2
        
        # Extract test data
        xtest = X_trans[test_start_index:]
        ytest = y_trans[test_start_index:]
        
        # Extract validation data
        xval = X_trans[val_start_index:test_start_index]
        yval = y_trans[val_start_index:test_start_index]
        
        # Remove test and validation data from X_trans and y_trans
        X_trans = X_trans[:val_start_index]
        y_trans = y_trans[:val_start_index]
    
        X_val, y_val = split_sequences(xval, yval, trainWindow, predWindow)
        print("Valid Shape:", X_val.shape, y_val.shape) 
    else:
        X_val,y_val,xtest,ytest=0,0,0,0
    
    X_train, y_train = split_sequences(X_trans, y_trans, trainWindow, predWindow)
    print("Training Shape:", X_train.shape, y_train.shape)
    

    return X_train,X_val,y_train,y_val,xtest,ytest,[ss,mm]

def getInd(df):
    rsi = getRSI(df)
    #df = getPSAR(df) #func already adds it to the df
    
    up,_,down = getBBAND(df,2,20)
    #add the indicator to the df
    df['rsi'] = rsi
    df['b_up'] = up
    df['b_dwn'] = down
    df = getHalving(df)
    return df

def getData(trainWindow,predWindow):
    #Read / calculate indicators / plot
    for coin in coins:
        df = readData(coin)
        df = getInd(df)
        dframes.append(df)
        #plotData(df, coin)
    
    #Join all coins data
    xtest,ytest,scalers = [],[],[]
    X_train,X_valid,y_train,y_valid = [],[],[],[]
    
    for df in dframes:
        df = formatData(df.to_numpy(),trainWindow,predWindow)
        X_train.append(df[0])
        X_valid.append(df[1])
        y_train.append(df[2])
        y_valid.append(df[3])
        xtest.append(df[4])
        ytest.append(df[5])
        scalers.append(df[6])
    
    X_train = np.concatenate(X_train, axis=0)
    X_valid = np.concatenate(X_valid, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    return X_train,y_train,X_valid,y_valid,xtest,ytest,scalers

class CustomDataset(Dataset):
    def __init__(self, data,lbl):
        #copy the data
        self.data = copy.deepcopy(data)
        self.lbl = copy.deepcopy(lbl)
        #load the given tokenizer
    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)
    def __getitem__(self, idx):
        return {'inp':torch.tensor(self.data[idx],dtype=torch.float32),
                'lbl':torch.tensor(self.lbl[idx],dtype=torch.float32),
                't':0}