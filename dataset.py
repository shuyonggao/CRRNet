#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset



########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, edge=None, ):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255, edge / 255


class RandomCrop(object):
    def __call__(self, image, mask=None, edge=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, edge=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), edge[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, edge=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        return image, mask, edge


class ToTensor(object):
    def __call__(self, image, mask=None, edge=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        edge = torch.from_numpy(edge)
        return image, mask, edge



########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        # datapath:
        self.cfg       = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        with open(self.cfg.datapath+'/'+ cfg.mode +'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        name = name[:-4]
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)

        if self.cfg.mode=='train':
            mask = cv2.imread(self.cfg.datapath + '/DUTS-TR-Mask/' + name + '.png', 0).astype(np.float32)
            edge = cv2.imread(self.cfg.datapath + '/DUTS-TR-Mask-Edge-Thick/' + name + '_edgedilate.png', 0).astype(np.float32)
            image, mask, edge = self.normalize(image, mask, edge)
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)
            return image, mask, edge

        else:
            shape = image.shape[:2]
            image= self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)
            return image, shape, name


    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask, edge


########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='datapath')
    data = Data(cfg)
    from torch.utils.data import DataLoader

    loader = DataLoader(data, collate_fn=data.collate, batch_size=1, shuffle=True, num_workers=8)
    for step, (image, mask, edge) in enumerate(loader):
        image, mask, edge= data[0]
        image = image * cfg.std + cfg.mean
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(np.uint8(image))
        plt.subplot(132)
        plt.imshow(mask)
        plt.subplot(133)
        plt.imshow(edge)
        plt.show()
        input()