# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as sio



def MaxMinNormalization(x):

    x = x.detach().cpu().numpy()
    Min = np.max(x)
    Max = np.min(x)
    x = (x - Min)/(Max - Min)
    x = torch.tensor(x)
    return x

def Inverse_covariance_S(Label, adj):
    LabelT = Label.transpose(0,-1)
    x = torch.matmul(Label,LabelT)
    x_norm = MaxMinNormalization(x)
    x_norm = torch.sqrt(x_norm)
    x_trace = torch.trace(x_norm)
    s = torch.div(x_norm,x_trace).cuda(0)
    s = adj.mul(s)
    s_row = s.sum(dim=1,keepdims=True)
    s_row[s_row==0] = 1
    s = s/s_row
    return s

class MLP(nn.Module):
    ## MLP
    def __init__(self,in_dims,out_dims):
        super(MLP, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.layer = nn.Linear(self.in_dims,self.out_dims)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1./math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv , stdv)
                m.bias.data.zero_()

    def forward(self,x):
        x = x.reshape(-1, self.in_dims)
        x = x.type_as(self.layer.weight.data)
        x_hat = self.layer(x)
        x_hat = F.relu(x_hat)
        return x_hat


class GAC(nn.Module):

    def __init__(self, in_features, out_features):
        super(GAC, self).__init__()
        self.in_dims = in_features
        self.out_dims = out_features
        self.MLP = MLP(self.in_dims,self.out_dims)

    def forward(self,Label,S):
        new_Label = torch.ones(self.out_dims,1).cuda(0)
        MLP_Label_new = self.MLP(Label)
        MLP_Label_new = MLP_Label_new.view(-1,63)
        for i in range(0,63):
            neighbor_to_i_relation = S[i,:].reshape(-1,63)
            n_to_i = torch.mul(neighbor_to_i_relation, MLP_Label_new)
            label_i = n_to_i.sum(dim = 1,keepdims=True)
            new_Label = torch.cat((new_Label, label_i),dim=1)
        new_Label = new_Label.T
        new_Label = new_Label[torch.arange(new_Label.size(0))!=0]

        return new_Label


class LGCN(nn.Module):

    def __init__(self,in_features, out_features,adj_dir):
        super(LGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        adj = sio.loadmat(adj_dir)['adj_less']
        self.adj = torch.from_numpy(adj).cuda(0)
        self.gc1 = GAC(self.in_features, 1024)
        self.gc2 = GAC(1024,self.out_features)


    def forward(self, labelgcn):
        labelgcn = torch.mean(labelgcn,dim = 0)
        S1 = Inverse_covariance_S(labelgcn, self.adj)
        S1 = S1.cuda(0)
        label1 = self.gc1(labelgcn,S1)
        S2 = Inverse_covariance_S(label1, self.adj)
        S2 = S2.cuda(0)
        label2 = self.gc2(label1,S2)

        return label2