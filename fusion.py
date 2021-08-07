# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_similarity(h, Label):
    beta = F.cosine_similarity(h,Label,dim=2)
    return beta


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, atte_d_model, d_u, d_c):
        super(MultiHeadAttention,self).__init__()

        self.d_model = atte_d_model
        self.d_u = d_u
        self.d_c = d_c
        self.n_head = self.d_model/self.d_c
        self.wq = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty(self.d_model,self.d_u), gain=nn.init.calculate_gain('relu')))
        self.wk = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty(self.d_model,self.d_c), gain=nn.init.calculate_gain('relu')))
        self.wv = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty(self.d_model, self.d_u), gain=nn.init.calculate_gain('relu')))
        self.wo = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty(1, self.d_model), gain=nn.init.calculate_gain('relu')))


    def forward(self, h,Label,batch_size):
        Label = Label.float()
        beta = cos_similarity(h,Label)
        h_ = torch.matmul(self.wq,h.permute(0,2,1))
        beta_ = torch.matmul(self.wk,beta.reshape(batch_size,-1,1))
        Label_ = torch.matmul(self.wv,Label.permute(0,2,1))
        H_to_Label = torch.matmul(h_,beta_.permute(0,2,1))

        head = torch.matmul(H_to_Label, Label_)
        y_hat = torch.matmul(self.wo, head)
        return y_hat

# model = MultiHeadAttention(126,512,63)
# print(model.state_dict())