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

#Link to dowload historical crypto data
#https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2=1711670400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

#PARAMS
path = 'S:/Cryp-Tor/'
#path = 'C:/Users/joels/Documents/GitHub/Cryp-Tor/'
device = 'cuda'
w=False
trainWindow = 256
predWindow = 1
batch = 32
epochs = 1
lr = 0.00008
inSize = 5
hidSize = 1024
outSize = predWindow
heads = 8 #bert has 12, large has 16
layers = 12 #bert has 12, large has 24
dout=0.4
criterion = nn.MSELoss()
coins=['BTC','ETH','XRP']
dframes = []
indicators = {'rsi':[]}
#Wandb

if w:
    wandb.init(
        # set the wandb project where this run will be logged
        name='repmode',
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

#Read / calculate indicators / plot
for coin in coins:
    df = readData(coin)
    rsi = getRSI(df)
    #add the indicator to the df
    df['rsi'] = rsi
    dframes.append(df)
    plotData(df, coin)

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
    
    X_train, y_train = split_sequences(X_trans, y_trans, trainWindow, predWindow)
    X_val, y_val = split_sequences(xval, yval, trainWindow, predWindow)
    
    print("Training Shape:", X_train.shape, y_train.shape)
    print("Valid Shape:", X_val.shape, y_val.shape) 

    return X_train,X_val,y_train,y_val,xtest,ytest,[ss,mm]


#Join all coins data
xtest,ytest,scalers = [],[],[]
X_train,X_valid,y_train,y_valid = [],[],[],[]

for df in dframes:
    df = formatData(df)
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
                'lbl':torch.tensor(self.lbl[idx],dtype=torch.float32),
                't':0}

'''
Model 
'''

class CrypTor(nn.Module):
    def __init__(self, inSize, hidSize, outSize, heads, layers, trainWindow,
                 dout=0.4, device='cpu'):
        super(CrypTor,self).__init__()
        self.encoder = nn.Linear(inSize, hidSize)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidSize, nhead=heads, batch_first=True)
        self.autobot = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(hidSize, outSize)
        self.dropout = nn.Dropout(dout)
        self.down = nn.Linear(256,1)
        self.mish = nn.Mish(inplace=True)
        
        #repmode
        self.num_experts=5
        self.num_tasks=1
        self.device=device
        self.repmode = MoDESubNetConv1d(self.num_experts,
                                        self.num_tasks,
                                        trainWindow,
                                        trainWindow)
        
    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, self.num_tasks))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)
        
    def forward(self,x,t):
        #[b,256,4]
        
        task_emb = self.one_hot_task_embedding(t)
        x = self.repmode(x,task_emb)
        #[b,256,64]

        x = self.encoder(x)
        x = self.mish(x)
        x = self.dropout(x)
        
        x = self.autobot(x)
        x = self.mish(x)
        x = self.dropout(x)
        #[b,256,64]
        x = self.down(x.permute([0,2,1])).squeeze(2)
        x = self.mish(x)
        #[b,64]
        x = self.fc(x)  
        #[b,1]
        return x
    
class MoDESubNetConv1d(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv2d(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv2d(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = x.unsqueeze(3) # [b,256,4,1]
        x = self.conv1(x, t) #[b,16,64,64,64] #[...,32,32,32] ... 4,4,4
        #x = self.rrdb(x)
        x = self.conv2(x, t)
        x = x.squeeze(3) # [b,256,4]
        return x

class MoDEConv2d(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                nn.BatchNorm2d(out_chan, affine=True), #torch.nn.BatchNorm3d(out_chan),
                nn.Mish(inplace=True)
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)


    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode='fan_out')
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K).mul(1.0 / K ** 2)
        return weight

    def trans_kernel(self, kernel, target_size):
        Hp = (target_size - kernel.shape[2]) // 2
        Wp = (target_size - kernel.shape[3]) // 2
        return F.pad(kernel, [Wp, Wp, Hp, Hp])

    def routing(self, g, N):

        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oihw,hw->oihw', self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum('oihw,hw->oihw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = list()
        for n in range(N):
            weight_nth_sample = torch.einsum('oihw,o->oihw', expert_conv5x5, g[n, 0, :]) + \
                                torch.einsum('oihw,o->oihw', expert_conv3x3, g[n, 1, :]) + \
                                torch.einsum('oihw,o->oihw', expert_conv1x1, g[n, 2, :]) + \
                                torch.einsum('oihw,o->oihw', expert_avg3x3, g[n, 3, :]) + \
                                torch.einsum('oihw,o->oihw', expert_avg5x5, g[n, 4, :])
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):

        N = x.shape[0] #batch size

        g = self.gate(t) #[b, x out channels * experts]
        g = g.view((N, self.num_experts, self.out_chan)) #[b,experts,x out channels]
        g = self.softmax(g)

        w = self.routing(g, N) #[b,x out chann, 1, 5,5,5] mix expert kernels

        if self.training:
            y = list()
            for i in range(N):
                y.append(F.conv2d(x[i].unsqueeze(0), w[i], bias=None, stride=1, padding='same'))
            y = torch.cat(y, dim=0)
        else:
            y = F.conv2d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y

'''
Training
'''

#Datasets
train_ds = CustomDataset(X_train,y_train)
test_ds = CustomDataset(X_valid, y_valid)
#Dataloaders
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True,pin_memory=True)
test_dl = DataLoader(test_ds,batch_size=batch,shuffle=False)
#Model and optim
model = CrypTor(inSize, hidSize, outSize, heads, layers,trainWindow,dout,device=device)
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
        t = data['t']
        
        pred = model(inp,t)
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
            t = data['t']
            pred = bestModel(inp,t)
            p.append(pred.cpu().detach())
            l.append(lbl.cpu().detach())
       
    pred = torch.cat(p,dim=0)
    lbl = torch.cat(l,dim=0)        
    pred_np = pred.detach().numpy() #[b,outsize]
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
        price = price/num[k] #average out predictions

    plt.plot(finalLbl, label='Actual Data') # actual plot
    plt.plot(finalPred, label='Predicted Data') # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.savefig(path + f"plots/{coins[i]}_whole_plot.png", dpi=300)
    plt.show() 
    
    # plt.plot(lbl_rv, label='Actual Data') # actual plot
    # plt.plot(pred_rv, label='Predicted Data') # predicted plot
    # plt.title('Time-Series Prediction')
    # plt.legend()
    # plt.savefig(path + f"plots/{coins[i]}_whole_plot.png", dpi=300)
    # plt.show() 