# coding=utf-8
import torch
from labelSpace import *
from multimodalRelation import *
from fusion import *
from utils import *
from torch.autograd import Variable
from collections import OrderedDict

class MMML(object):
    """
    uvs model
    """
    def __init__(self, opt):
        ## Build Models
        super(MMML, self).__init__()
        self.opt = opt
        self.subLearning = SubspaceLearning(opt.sl_in_dim, opt.sl_hidden_dim, opt.sl_out_dim, opt.fusion_style, opt.sl_dropout)
        self.AE = AutoEncoder(opt.dim_encoder, opt.out_dim_common)

        self.lgcn = LGCN(opt.lgcn_in_features, opt.lgcn_out_features, opt.adj)
        self.multiAttn = MultiHeadAttention(opt.atte_d_model, opt.d_u, opt.d_c, opt.fusion_dropout)
        params = list(self.subLearning.parameters())
        params += list(self.AE.parameters())
        params += list(self.lgcn.parameters())
        params += list(self.multiAttn.parameters())
        self.params = params

        if opt.optimizer.type == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer.type == "SGD":
            self.optimizer = torch.optim.SGD(params, lr = opt.learning_rate,
                                             momentum = opt.optimizer.momentum,
                                             weight_decay = opt.optimizer.weight_decay,
                                             nesterov = opt.optimizer.nesterov)
        else:
            raise NotImplementedError('Only support Adam and SGD optimizer.')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, opt.lr_scheduler.lr_steps,gamma= 0.1,last_epoch=-1)

        if torch.cuda.is_available():
            self.subLearning.cuda(0)
            self.AE.cuda(0)
            self.lgcn.cuda(0)
            self.multiAttn.cuda(0)

        self.Eiters = 0

    def recon_clas_criterion(self, x_hat,x,y_hat,y, smi_loss):
        recon_loss = F.mse_loss(x_hat,x)
        ## class loss
        classification = torch.nn.BCEWithLogitsLoss()
        y = y.squeeze(1)
        c_loss = classification(y_hat, y.float())
        all_loss = self.opt.lam1 * c_loss + self.opt.lam2 * recon_loss + self.opt.lam3 * smi_loss

        return all_loss

    def state_dict(self):
        state_dict =[self.subLearning.state_dict(),self.AE.state_dict(),self.lgcn.state_dict(),self.multiAttn.state_dict()]
        return state_dict

    def load_state_dict(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.subLearning.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            new_state_dict[k] = v
        self.AE.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[2].items():
            new_state_dict[k] = v
        self.lgcn.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[3].items():
            new_state_dict[k] = v
        self.multiAttn.load_state_dict(new_state_dict, strict=True)



    def train_start(self):
        """
        switch to the train mode
        :return:
        """
        self.subLearning.train()
        self.AE.train()
        self.lgcn.train()
        self.multiAttn.train()

    def eval_start(self):
        """
        switch to the eval mode
        :return:
        """
        self.subLearning.eval()
        self.AE.eval()
        self.lgcn.eval()
        self.multiAttn.eval()

    def train_emb(self,vision,audio,tra,y_true,labelgcn,batch_size):
        vision = Variable(vision)
        audio = Variable(audio)
        tra = Variable(tra)
        labelgcn = Variable(labelgcn)
        y_true = Variable(y_true)
        if torch.cuda.is_available():
            vision = vision.cuda(0)
            audio = audio.cuda(0)
            tra = tra.cuda(0)
            labelgcn = labelgcn.cuda(0)
            y_true = y_true.cuda(0)
        self.optimizer.zero_grad()
        _,h2, h3, h4 = self.subLearning(vision, audio,tra)
        sim_loss = (mmatch(h2,h3,3) + mmatch(h2,h4,3) + mmatch(h3,h4,3))/3
        h_ = torch.cat((h2, h3, h4), dim=1)
        H_common, x_hat = self.AE(h_)
        vision_new = H_common + vision
        Label = self.lgcn(labelgcn)
        Label = Label.reshape(1, 63, -1).repeat(batch_size, 1, 1)
        vision_new = vision_new.reshape(batch_size, 1, -1)
        y_hat = self.multiAttn(vision_new, Label, batch_size).squeeze(1)

        loss = self.recon_clas_criterion(x_hat, h_,  H_common,vision, y_hat, y_true, sim_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.params,self.opt.grad_clip)
        self.optimizer.step()
        return y_hat, loss


    def test_emb(self,vision,audio,tra,y_true,labelgcn,batch_size):
        vision = Variable(vision)
        audio = Variable(audio)
        tra = Variable(tra)
        labelgcn = Variable(labelgcn)
        y_true = Variable(y_true)
        if torch.cuda.is_available():
            vision = vision.cuda(0)
            audio = audio.cuda(0)
            tra = tra.cuda(0)
            labelgcn = labelgcn.cuda(0)
            y_true = y_true.cuda(0)
        with torch.no_grad():
            _, h2, h3, h4 = self.subLearning(vision, audio, tra)
            sim_loss = (mmatch(h2, h3, 3) + mmatch(h2, h4, 3) + mmatch(h3, h4, 3)) / 3
            h_ = torch.cat((h2, h3, h4), dim=1)
            H_common, x_hat = self.AE(h_)
            vision_new = H_common + vision
            Label = self.lgcn(labelgcn)
            Label = Label.reshape(1, 63, -1).repeat(batch_size, 1, 1)
            vision_new = vision_new.reshape(batch_size, 1, -1)
            y_hat = self.multiAttn(vision_new, Label, batch_size).squeeze(1)

            loss = self.recon_clas_criterion(x_hat, h_, H_common, vision, y_hat, y_true, sim_loss)
        return y_hat, loss

