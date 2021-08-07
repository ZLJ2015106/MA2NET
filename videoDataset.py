# coding=utf-8
from torch.utils.data import Dataset
import scipy.io as sio
class VideoDataset(Dataset):
    def __init__(self,label_dir,visual_dir,audio_dir,tra_dir,labelgcn_name,split):
        self.split = split
        self.visual = sio.loadmat(visual_dir)[self.split]
        self.audio = sio.loadmat(audio_dir)[self.split]
        self.tra = sio.loadmat(tra_dir)[self.split]
        self.label = sio.loadmat(label_dir)[self.split]

        self.labelgcn = sio.loadmat(labelgcn_name)['labelVector']

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,idx):
        visual = self.visual[idx]
        audio = self.audio[idx]
        tra = self.tra[idx]
        label = self.label[idx]
        return visual, audio, tra, label, self.labelgcn


if  __name__ == "__main__":
    from torch.utils.data import DataLoader
    label_dir="/media/Harddisk/zlj/data/Label/Y_test.mat"
    visual_dir="/media/Harddisk/zlj/data/vision/vision_test.mat"
    audio_dir="/media/Harddisk/zlj/data/audio/mfcc_test.mat"
    tra_dir = "/media/Harddisk/zlj/data/tra/tra_test.mat"
    train_data = VideoDataset(label_dir,visual_dir,audio_dir,tra_dir,labelgcn_name='/media/Harddisk/zlj/dataset/label.mat', split = 'test')
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4)
    for i,sample in enumerate(train_loader):
        visual = sample[0]
        audio = sample[1]
        tra = sample[2]
        label_info = sample[3]
        print(visual.shape)
        print(audio.shape)
        print(tra.shape)
        print(label_info.shape)
        # print(label_info)
        if i == 1:
            break