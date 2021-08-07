# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import math
from utils import *


def complementary_study(v,other):
    v = v.transpose(0,1)
    R = torch.matmul(v,other)
    Rc = F.softmax(R,dim = 1)
    other = other.transpose(0,-1)
    other_to_vision = torch.matmul(Rc,other)
    other_to_vision = other_to_vision.transpose(0, -1)
    return other_to_vision


class SubspaceLearning(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, fusion_style = 'concat'):
        super(SubspaceLearning,self).__init__()
        self.in_dim = in_dim
        self.fusion_style = fusion_style
        if self.fusion_style == "concat":
            self.add_dim = 4096
        else:
            self.add_dim = in_dim

        self.out_dim = out_dim
        self.fusion_style = fusion_style
        self.hidden_dim = hidden_dim

        self.fc_1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(self.add_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.Tanh(),
            nn.Linear(self.in_dim, self.out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
                 )
        self.fc_3 = nn.Sequential(
            nn.Linear(self.add_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.Tanh(),
            nn.Linear(self.in_dim, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.Tanh()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear(self.add_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.Tanh(),
            nn.Linear(self.in_dim, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            if isinstance(m,nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,v,a,tra):
        h1 = self.fc_1(v)
        a_hat = complementary_study(v, a)
        t_hat = complementary_study(v, tra)
        a_t = torch.cat((a, tra), dim=1)
        a_t_hat = complementary_study(v, a_t)
        if self.fusion_style == "sum":
            v_hat_2 = v + a_hat
            v_hat_3 = v + t_hat
            v_hat_4 = v + a_t_hat
        elif self.fusion_style == "concat":
            v_hat_2 = torch.cat((v, a_hat), dim=1)
            v_hat_3 = torch.cat((v, t_hat), dim=1)
            v_hat_4 = torch.cat((v, a_t_hat), dim=1)
        h2 = self.fc_2(v_hat_2)
        h3 = self.fc_3(v_hat_3)
        h4 = self.fc_4(v_hat_4)
        return h1, h2, h3, h4

class AutoEncoder(nn.Module):
    def __init__(self,dim_encoder,out_dim_common):
        super(AutoEncoder,self).__init__()
        self.dim = dim_encoder
        self.out_dim = out_dim_common
        self.hidden = (self.out_dim + self.dim)//2
        self.encoder = nn.Sequential(
            nn.Linear(self.dim, self.out_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.dim),
            nn.ReLU(),
        )
        self.weights = self.init_weights()
    #
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                m.bias.data.zero_()

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# model = SubspaceLearning(2048,1024,900,732)
# print(model.state_dict())
# model = AutoEncoder(732,512)
# print(model.state_dict())