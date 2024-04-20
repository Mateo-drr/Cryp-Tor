# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:33:02 2024

@author: Mateo-drr
"""
import dataPrep as prep
import numpy as np

def cryptorLink(coinData,model,device):
    coinData = prep.getInd(coinData)
    coinData = prep.formatData(coinData,split=False)
    data,lbl = coinData[0],coinData[2]
    data = np.concatenate((data, lbl))
    ds = prep.CustomDataset(data, [0])
    #dl = DataLoader(ds,batch_size=1,shuffle=False,pin_memory=True)
    
    inp = data['inp'].to(device)
    t = data['t']
    pred = model(inp,t)
    return pred