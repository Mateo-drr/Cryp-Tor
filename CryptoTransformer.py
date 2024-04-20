# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:50:52 2022

@author: Mateo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import copy
import wandb
from indicators.rsi import getRSI
import math
from indicators.psar import getPSAR
from indicators.bollingerband import getBBAND
from indicators.halving import getHalving
from tqdm import tqdm
import torch.nn.utils as torch_utils
import dataPrep as prep
from model import CrypTor

#Open high low close volume
#Link to dowload historical crypto data
#https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2=1711670400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

#PARAMS
path = 'S:/Cryp-Tor/'
#path = 'C:/Users/joels/Documents/GitHub/Cryp-Tor/'
version = 'F'
device = 'cuda'
w=True
trainWindow = 256
predWindow = 1 
batch = 32
epochs = 500
lr = 0.00005
clip=3 #grad clipping
inSize = 9
hidSize = 1256
outSize = predWindow
heads = 8 #bert has 12, large has 16
layers = 12 #bert has 12, large has 24
dout=0.1
criterion = nn.MSELoss()
coins=['BTC',
       'ETH',
       'XRP',
       'BNB',
       'DOGE',
       'ADA',
       'DOT',
       'SOL',
       ]
dframes = []
indicators = {'rsi':[]}
plots=['Open','High','Low','Close','Volume','RSI','BolUP','BolDW','Halving']

torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark=True

#Wandb
if w:
    wandb.init(
        # set the wandb project where this run will be logged
        name=f'Cryp-Tor v{version}',
        project="Cryp-Tor",
        entity='unitnais',
    
        # track hyperparameters and run metadata
        config={
        "train_window": trainWindow,
        "pred_window": predWindow,
        "batch_size": batch,
        "epochs": epochs,
        "learning_rate": lr,
        "input_size": inSize,
        "hidden_size": hidSize,
        "output_size": outSize,
        "num_heads": heads,
        "num_layers": layers,
        'coins':coins,
        'comments':'No p-sar'
        }
    )

X_train,y_train,X_valid,y_valid,xtest,ytest,scalers = prep.getData(trainWindow,predWindow)

###############################################################################
#PYTORCH
###############################################################################

def smape(actual, forecast):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Parameters:
        actual (tensor): Tensor of actual values
        forecast (tensor): Tensor of forecasted values
    
    Returns:
        tensor: SMAPE value
    """
    denominator = (torch.abs(actual) + torch.abs(forecast)) / 2
    diff = torch.abs(actual - forecast) / denominator
    diff[denominator == 0] = 0  # Handle division by zero
    smape_val = torch.mean(diff) * 100
    return smape_val



'''
Training
'''

#criterion = smape

#Datasets
train_ds = prep.CustomDataset(X_train,y_train)
test_ds = prep.CustomDataset(X_valid, y_valid)
#Dataloaders
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True,pin_memory=True)
test_dl = DataLoader(test_ds,batch_size=batch,shuffle=False)
#Model and optim
model = CrypTor(inSize, hidSize, outSize, heads, layers,trainWindow,
                predWindow,dout=dout,device=device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#LOOPS
bestL = 1e20
for epoch in range(epochs):
    #Train
    model.train()
    t_loss = 0
    
    for data in tqdm(train_dl):
        inp = data['inp'].to(device)
        lbl = data['lbl'].to(device)
        t = data['t']
        
        pred = model(inp,t)
        loss = criterion(pred,lbl)
        
        optimizer.zero_grad()
        loss.backward()
        torch_utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        t_loss += loss
        
    model.eval()
    v_loss = 0
    with torch.no_grad():    
        for data in test_dl:
            inp = data['inp'].to(device)
            lbl = data['lbl'].to(device)
            t = data['t']
            
            pred = model(inp,t)
            loss = criterion(pred,lbl)
            v_loss += loss
    
    print(f'E{epoch+1}: T:{t_loss/len(train_dl)} V:{v_loss/len(test_dl)}')
    if w:
        wandb.log({"train loss": t_loss/len(train_dl), "valid loss": v_loss/len(test_dl)})
    if v_loss <= bestL:
        bestL = v_loss
        bestModel = copy.deepcopy(model)
        e = epoch

torch.cuda.empty_cache()
print('Best model on epoch', e+1, 'L:', bestL/len(test_dl))
torch.save(bestModel.state_dict(), path+f'CTv{version}_{e}')

#loop the coins used to train
for i in range(len(xtest)):
    #Data for plotting
    
    #skip coins with insufficient data
    if len(xtest[i])< trainWindow:
        continue
    
    x,y = prep.split_sequences(xtest[i], ytest[i], trainWindow, predWindow)
    tds = prep.CustomDataset(x,y)
    tdl = DataLoader(tds,batch_size=batch)
    
    #Predict
    p,l = [],[]
    with torch.no_grad():
        for data in tdl:
            inp = data['inp'].to(device)
            lbl = data['lbl'].to(device)
            t = data['t']            
            pred = bestModel(inp,t)
            p.append(pred.cpu().detach())
            l.append(lbl.cpu().detach())
       
    pred = torch.cat(p,dim=0)
    lbl = torch.cat(l,dim=0)        
    pred_np = pred.detach().numpy() #[b,outsize]
    lbl_np = lbl.numpy()
    
    #reverse re-scaling
    #mm = scalers[i][1]
    ss = scalers[i][0]
    #pred_rv_a = ss.inverse_transform(pred_np)#mm.inverse_transform(pred_np)
    #lbl_rv_a = ss.inverse_transform(lbl_np)#mm.inverse_transform(lbl_np)
    
    # train_predict = lstm(df_X_ss) # forward pass
    # data_predict = train_predict.data.numpy() # numpy conversion
    # dataY_plot = df_y_mm.data.numpy()
    
    # data_predict = mm.inverse_transform(data_predict) # reverse transformation
    # dataY_plot = mm.inverse_transform(dataY_plot)
    # true, preds = [], []
    # for i in range(len(dataY_plot)):
    #     true.append(dataY_plot[i][0])
    # for i in range(len(data_predict)):
    #     preds.append(data_predict[i][0])
    # plt.figure(figsize=(10,6)) #plotting
    # plt.axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set
    
    for idx in range(pred_np.shape[-1]):
        pred_rv = pred_np[:,:,idx]
        lbl_rv = lbl_np[:,:,idx]
        finalPred=[0]*(len(pred_rv)+outSize)
        num = [0]*(len(pred_rv)+outSize)
        finalLbl = []
        for k,window in enumerate(pred_rv):
            #Add price predictions of each window
            for j in range(0,outSize):    
                finalPred[j+k] += window[j]
                num[j + k] += 1
            
            #Get only the first price from the label
            finalLbl.append(lbl_rv[k][0])
    
        #remove extra index
        finalPred,num = finalPred[:-1],num[:-1]
        #add the last 4 values to the labels
        finalLbl += list(lbl_rv[-1][1:])
        
        for k,price in enumerate(finalPred):
            finalPred[k] = price/num[k] #average out predictions
        
        #finalLbl = ss.inverse_transform(finalLbl)
        #finalPred = ss.inverse_transform(finalPred)
    
        plt.plot(finalLbl, label='Actual Data') # actual plot
        plt.plot(finalPred, label='Predicted Data') # predicted plot
        plt.title(f'{coins[i]} {plots[idx]} Prediction')
        plt.legend()
        plt.savefig(path + f"plots/{coins[i]}_{plots[idx]}.png", dpi=300)
        plt.show() 
    
    # plt.plot(lbl_rv, label='Actual Data') # actual plot
    # plt.plot(pred_rv, label='Predicted Data') # predicted plot
    # plt.title('Time-Series Prediction')
    # plt.legend()
    # plt.savefig(path + f"plots/{coins[i]}_whole_plot.png", dpi=300)
    # plt.show() 