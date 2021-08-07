# coding=utf-8
import numpy as np
import torch


def avg_p(output, target):

    sorted, indices = torch.sort(output, dim=1, descending=True)
    tp = 0
    s = 0
    for i in range(target.size(1)):
        idx = indices[0,i]
        if target[0,idx] == 1:
            tp = tp + 1
            pre = tp / (i+1)
            s = s + pre
    if tp == 0:
        AP = 0
    else:
        AP = s/tp
    return AP

def cal_ap(y_pred,y_true):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    ap = torch.zeros(y_pred.size(0))
    # compute average precision for each class
    for k in range(y_pred.size(0)):
        # sort scores
        scores = y_pred[k,:].reshape([1,-1])
        targets = y_true[k,:].reshape([1,-1])
        ap[k] = avg_p(scores, targets)
    return ap

def cal_one_error(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    one_error = 0
    for i in range(num_instance):
        indicator = 0
        Label = []
        not_Label = []
        temp_tar = target[i, :].reshape(1, num_class)
        for j in range(num_class):
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i, :].cpu().numpy()
        maximum = max(temp_out)
        index = np.argmax(temp_out)
        for j in range(num_class):
            if (temp_out[j] == maximum):
                if index in Label:
                    indicator = 1
                    break
        if indicator == 0:
            one_error = one_error + 1

    one_error = one_error / num_instance
    return one_error

def cal_coverage(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    cover = 0
    for i in range(num_instance):
        Label = []
        not_Label = []
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):
            if(temp_tar[0,j]==1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i,:]
        _,inde = torch.sort(temp_out)
        inde = inde.cpu().numpy().tolist()
        temp_min = num_class
        for m in range(Label_size):
            loc = inde.index(Label[m])
            if (loc<temp_min):
                temp_min = loc
        cover = cover + (num_class-temp_min)

    cover_result = (cover/num_instance)-1
    return cover_result


def cal_RankingLoss(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    rankloss = 0
    for i in range(num_instance):
        Label = []
        not_Label = []
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp = 0
        for m in range(Label_size):
            for n in range(num_class-Label_size):
                if output[i,Label[m]]<=output[i,not_Label[n]]:
                    temp += 1
        if Label_size==0:
            continue
        else:
            rankloss = rankloss + temp / (Label_size * (num_class-Label_size))

    RankingLoss = rankloss / num_instance
    return RankingLoss

def cal_HammingLoss(output, target):
    pre_output = torch.zeros(output.size(0),output.size(1)).cuda(0)
    for i in range(output.size(0)):
        for j in range(output.size(1)):
            if output[i,j]>=0.5:
                pre_output[i,j]=1
            else:
                pre_output[i,j]=0
    num_class, num_instance = output.size(1), output.size(0)
    miss_sum = 0
    for i in range(num_instance):
        miss_pairs = sum(pre_output[i,:]!=target[i,:].cuda(0))
        miss_sum += miss_pairs
    HammingLoss = miss_sum/(num_class*num_instance)

    return HammingLoss

def matchnorm(x1,x2):
    return torch.sqrt(torch.sum(torch.pow(x1 - x2,2)))


def scm(sx1, sx2, k):
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return matchnorm(ss1,ss2)


def mmatch(x1,x2,n_moments):
    xx1 = torch.mean(x1,0)
    xx2 = torch.mean(x2,0)
    sx1 = x1 - xx1
    sx2 = x2 - xx2
    dm = matchnorm(xx1, xx2)
    scms = dm
    for i in range(n_moments-1):
        scms = scm(sx1, sx2, i+2) + scms
    return scms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)



