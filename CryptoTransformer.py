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

#Link to dowload historical crypto data
#https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2=1711670400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

#PARAMS
path = 'S:/Cryp-Tor/'
device = 'cuda'
trainWindow = 256
predWindow = 1
batch = 32
epochs = 25
lr = 0.0001
inSize = 4
hidSize = 64
outSize = 1
heads = 16 #bert has 12, large has 16
layers = 24 #bert has 12, large has 24
criterion = nn.MSELoss()

#Wandb
wandb.init(
    # set the wandb project where this run will be logged
    name='test3',
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
    "num_layers": layers
    }
)

def readData(name):
    df = pd.read_csv(path + f'{name}-USD.csv', index_col = 'Date', parse_dates=True)
    df.drop(columns=['Adj Close'], inplace=True)
    df.head(5)
    return df

def plotData(df,name):
    plt.plot(df.Close)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.savefig(path + f"{name}_initial_plot.png", dpi=250)
    plt.show();

df = readData('BTC')
df2 = readData('ETH')

plotData(df, 'BTC')
plotData(df2, 'ETH')

# split a multivariate sequence past, future samples (X and y)
# TLDR it organizes the data, so that for example you have to predict
# 3 days given 10 past days
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def formatData(df):
    X, y = df.drop(columns=['Close']), df.Close.values
    X.shape, y.shape
    
    #Re-scale the data 
    mm = MinMaxScaler(feature_range=(0,1)) #limit betwee 0 and 1
    ss = StandardScaler() #mean 0 and std 1
    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1))
    
    #Remove test data
    xtest = X_trans[-365:]
    ytest = y_trans[-365:]
    
    X_ss, y_mm = split_sequences(X_trans, y_trans, trainWindow, predWindow)
    print(X_ss.shape, y_mm.shape)

    #TODO possible mix of train data in validation?
    total_samples = len(X)
    train_test_cutoff = round(0.10 * total_samples)
    
    X_train = X_ss[:-train_test_cutoff]
    X_val = X_ss[-train_test_cutoff:]
    
    y_train = y_mm[:-train_test_cutoff]
    y_val = y_mm[-train_test_cutoff:] 
    
    print("Training Shape:", X_train.shape, y_train.shape)
    print("Valid Shape:", X_val.shape, y_val.shape) 

    return X_train,X_val,y_train,y_val,xtest,ytest,[ss,mm]

def joinData(d1,d2):
    X_train = np.concatenate((d1[0],d2[0]),axis=0)
    X_valid = np.concatenate((d1[1],d2[1]),axis=0)
    y_train = np.concatenate((d1[2],d2[2]),axis=0)
    y_valid = np.concatenate((d1[3],d2[3]),axis=0)
    xtest,ytest = [],[]
    xtest.append(d1[4])
    xtest.append(d2[4])
    ytest.append(d1[5])
    ytest.append(d2[5])
    return X_train,X_valid,y_train,y_valid,xtest,ytest,[d1[6],d2[6]]

X_train,X_valid,y_train,y_valid,xtest,ytest,scalers = joinData(formatData(df), formatData(df2))

###############################################################################
#PYTORCH
###############################################################################

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
                'lbl':torch.tensor(self.lbl[idx],dtype=torch.float32)}

class CrypTor(nn.Module):
    def __init__(self, inSize, hidSize, outSize, heads, layers, dout=0.1):
        super(CrypTor,self).__init__()
        self.encoder = nn.Linear(inSize, hidSize)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidSize, nhead=heads, batch_first=True)
        self.autobot = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(hidSize, outSize)
        self.dropout = nn.Dropout(dout)
        self.down = nn.Linear(256,1)
        
    def forward(self,x):
        #[b,256,4]
        x = self.encoder(x)
        #[b,256,64]
        x = self.dropout(x)
        x = self.autobot(x)
        x = self.dropout(x)
        #[b,256,64]
        x = self.down(x.permute([0,2,1])).squeeze(2)
        #[b,64]
        x = self.fc(x)  
        #[b,1]
        return x

#Datasets
train_ds = CustomDataset(X_train,y_train)
test_ds = CustomDataset(X_valid, y_valid)
#Dataloaders
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True,pin_memory=True)
test_dl = DataLoader(test_ds,batch_size=batch,shuffle=False)
#Model and optim
model = CrypTor(inSize, hidSize, outSize, heads, layers)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#LOOPS
bestL = 100
for epoch in range(epochs):
    #Train
    model.train()
    t_loss = 0
    
    for data in train_dl:
        inp = data['inp'].to(device)
        lbl = data['lbl'].to(device)
        
        pred = model(inp)
        loss = criterion(pred,lbl)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_loss += loss
        
    model.eval()
    v_loss = 0
    with torch.no_grad():    
        for data in test_dl:
            inp = data['inp'].to(device)
            lbl = data['lbl'].to(device)
            
            pred = model(inp)
            loss = criterion(pred,lbl)
            v_loss += loss
    
    print(f'E{epoch+1}: T:{t_loss/len(train_dl)} V:{v_loss/len(test_dl)}')
    wandb.log({"train loss": t_loss/len(train_dl), "valid loss": v_loss/len(test_dl)})
    if v_loss <= bestL:
        bestL = v_loss
        bestModel = copy.deepcopy(model)
        e = epoch

torch.cuda.empty_cache()
print('Best model on epoch', e, 'L:', bestL/len(test_dl))

#loop the coins used to train
for i in range(len(xtest)):
    #Data for plotting
    x,y = split_sequences(xtest[i], ytest[i], trainWindow, predWindow)
    tds = CustomDataset(x,y)
    tdl = DataLoader(tds,batch_size=batch)
    
    #Predict
    p,l = [],[]
    with torch.no_grad():
        for data in tdl:
            inp = data['inp'].to(device)
            lbl = data['lbl'].to(device)
            pred = bestModel(inp)
            p.append(pred.cpu().detach())
            l.append(lbl.cpu().detach())
       
    pred = torch.cat(p,dim=0)
    lbl = torch.cat(l,dim=0)        
    pred_np = pred.detach().numpy()
    lbl_np = lbl.numpy()
    
    #reverse re-scaling
    mm = scalers[i][1]
    pred_rv = mm.inverse_transform(pred_np)
    lbl_rv = mm.inverse_transform(lbl_np)
    
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
    
    plt.plot(lbl_rv, label='Actual Data') # actual plot
    plt.plot(pred_rv, label='Predicted Data') # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.savefig(path + "whole_plot.png", dpi=300)
    plt.show() 