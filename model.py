# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:35:09 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrypTor(nn.Module):
    def __init__(self, inSize, hidSize, outSize, heads, layers, trainWindow,
                 predWindow, dout=0.4, device='cpu'):
        super(CrypTor,self).__init__()
        self.encoder = nn.Linear(inSize, hidSize)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidSize, nhead=heads, batch_first=True)
        self.autobot = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(hidSize, inSize)
        self.dropout = nn.Dropout(dout)
        self.down = nn.Linear(trainWindow,predWindow)
        self.mish = nn.Mish(inplace=True)
        self.bn = nn.BatchNorm1d(hidSize,affine=True)
        #self.bn2 = nn.BatchNorm1d(hidSize,affine=True)
        
        #repmode
        self.num_experts=5
        self.num_tasks=1
        self.device=device
        self.repmode = MoDESubNetConv1d(self.num_experts,
                                        self.num_tasks,
                                        trainWindow,
                                        trainWindow)
        # self.repmode0 = MoDESubNetConv1d(self.num_experts,
        #                                 self.num_tasks,
        #                                 trainWindow,
        #                                 trainWindow)
        
    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, self.num_tasks))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)
        
    def forward(self,x,t):
        #[b,256,9]
        task_emb = self.one_hot_task_embedding(t)
        #x = self.repmode0(x,task_emb)
        
        x = self.encoder(x)
        #x = self.bn(x.permute([0,2,1])).permute([0,2,1]) #normalize features
        x = self.mish(x)
        x = self.dropout(x)
        #[b,256,hidSize]
        
        
        x = self.repmode(x,task_emb)
        xj = self.dropout(x)
        #[b,256,hidSize]
        
        x = self.autobot(xj)
        #x = self.bn(x.permute([0,2,1])).permute([0,2,1]) #normalize features
        x = self.mish(x)
        x = self.dropout(x) + 0.2*xj
        #[b,256,hidSize]
        
        x = self.down(x.permute([0,2,1])).permute([0,2,1])
        #x = self.bn(x)
        x = self.mish(x)
        x = self.dropout(x)
        #[b,hidSize]
        
        x = self.fc(x)  
        #[b,1]
        
        return x
    
class MoDESubNetConv1d(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv2d(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same')
        #self.conv2 = MoDEConv2d(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = x.unsqueeze(3) # [b,256,4,1]
        x = self.conv1(x, t) #[b,16,64,64,64] #[...,32,32,32] ... 4,4,4
        #x = self.rrdb(x)
        #x = self.conv2(x, t)
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
                #nn.BatchNorm2d(out_chan, affine=True), #torch.nn.BatchNorm3d(out_chan),
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
