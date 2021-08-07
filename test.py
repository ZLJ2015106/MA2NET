# coding = utf-8
import torch
from utils import *
from model import *
from videoDataset import VideoDataset
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict

def test():
    with open(r"/home/zlj/MCSRv_Final/args.yaml") as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['common'])
    torch.manual_seed(2)
    ## read data
    testdataset = VideoDataset(opt.test_label_dir, opt.test_visual_dir, opt.test_audio_dir, opt.test_tra_dir,
                               opt.labelgcn_name, split='test')
    test_loader = DataLoader(testdataset, batch_size=opt.batch_size, shuffle=True, num_workers=4,drop_last=True)

    model = MMML(opt)
    ## load model parameters and print the information
    print("=> loading checkpoint '{}'".format(opt.resume))
    checkpoint = torch.load(opt.resume)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, best_accuracy {})"
          .format(opt.resume, start_epoch, best_accuracy))

    running_loss = 0
    map = 0
    coverage = 0
    rankingLoss = 0
    HammingLoss =0
    one_error = 0

    with torch.no_grad():
        print("start validate")
        print("start loading val data...")
        model.eval_start()
        for i, test_data in enumerate(test_loader):
            y_hat, loss = model.test_emb(*test_data, opt.batch_size)
            running_loss = loss + running_loss
            map_batch = cal_ap(y_hat, test_data[3]).mean()
            coverage_batch = cal_coverage(y_hat, test_data[3])
            rankingLoss_batch = cal_RankingLoss(y_hat, test_data[3])
            HammingLoss_batch = cal_HammingLoss(y_hat, test_data[3])
            one_error_batch = cal_one_error(y_hat, test_data[3])
            map = map + map_batch
            coverage = coverage + coverage_batch
            rankingLoss = rankingLoss + rankingLoss_batch
            HammingLoss = HammingLoss + HammingLoss_batch
            one_error = one_error + one_error_batch
            # Record logs in tensorboard
            print('Test: [{0}/{1}]\t'
                  'mAP_batch {map_batch: .4f}\t'
                  'coverage_batch {coverage_batch: .4f}\t'
                  'rankingLoss_batch {rankingLoss_batch: .4f}\t'
                  'HammingLoss_batch {HammingLoss_batch: .4f}\t'
                  'one_error_batch {one_error_batch: .4f}'.format(
                  i, len(test_loader), map_batch=map_batch, coverage_batch = coverage_batch, rankingLoss_batch = rankingLoss_batch,
                  HammingLoss_batch = HammingLoss_batch, one_error_batch = one_error_batch))
    map = map / len(test_loader)
    coverage = coverage / len(test_loader)
    rankingLoss = rankingLoss / len(test_loader)
    HammingLoss = HammingLoss / len(test_loader)
    one_error = one_error / len(test_loader)
    print('Test mAP {map:.4f} \t'
          'coverage {coverage:.4f}\t'
          'rankingLoss {rankingLoss:.4f}\t'
          'HammingLoss {HammingLoss:.4f}\t'
          'one_error {one_error:.4f}'.format(map=map, coverage=coverage, rankingLoss=rankingLoss, HammingLoss=HammingLoss,
                                             one_error=one_error))

if __name__ == '__main__':
    test()
