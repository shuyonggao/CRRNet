#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
from net_v2 import CRRN

class Test(object):
    def __init__(self, Dataset, Network, Path, snapshot):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=snapshot, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net = nn.DataParallel(self.net)

        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
        self.net.train(False)
    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:

                image, shape = image.cuda().float(), (H, W)

                up_edge, up_sal, output = self.net(image)
                ##--------------------------show edge and sal---------------------##
                import matplotlib.pyplot as plt
                import cv2
                edge = up_edge[1].squeeze(0)
                edge = edge.data.cpu().numpy()
                print(edge.shape)


                plt.imshow(edge[0], cmap='gray')
                plt.show()

                ##----------------------------------------------------------##

                out = output
                pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                pred = cv2.resize(pred, dsize=(W,H), interpolation=cv2.INTER_LINEAR)
                head = './eval/maps/CRRNet/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))


if __name__=='__main__':
    for path in ['/home/gaosy/DATA/GT/ECSSD', '/home/gaosy/DATA/GT/PASCAL_S', '/home/gaosy/DATA/GT/DUTS_test',
                 '/home/gaosy/DATA/GT/HKU_IS', '/home/gaosy/DATA/GT/DUT_O', '/home/gaosy/DATA/GT/SOD']:
        t = Test(dataset, CRRN, path, './out/'+'model-45')
        t.save()

