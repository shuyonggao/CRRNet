#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net_v2 import CRRN
import torchvision.utils as vutils


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters: {}".format(num_params))

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def train(Dataset, Network):
    # dataset
    cfg = Dataset.Config(datapath='datapath', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=45)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)

    # network
    net =  Network(cfg)  # cfg
    net = nn.DataParallel(net)
    net.train(True)
    net.cuda()

    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():

        if 'base' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    print_network(net, 'CRRN')
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step , (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.type(torch.FloatTensor).cuda(), mask.type(torch.FloatTensor).cuda(), edge.type(torch.FloatTensor).cuda()
            out_edge, out_sal, out_final = net(image)

            edge_loss,sal_loss, final_loss = [], [], []

            num_edg = 0
            for edge_i in out_edge:
                edge_loss.append(F.binary_cross_entropy_with_logits(edge_i, edge))
                num_edg += 1
            edge_loss = sum(edge_loss)/num_edg

            num_sal=0
            for sal_i in out_sal:
                sal_loss.append(F.binary_cross_entropy_with_logits(sal_i, mask))
                num_sal += 1
            sal_loss = sum(sal_loss)/num_sal

            final_loss = F.binary_cross_entropy_with_logits(out_final, mask) + iou_loss(out_final,mask)
            loss = (edge_loss + sal_loss + final_loss)/2

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'edge_loss':edge_loss.item(), 'sal_loss':sal_loss.item(), 'final_loss':final_loss.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | edge_loss=%.6f | sal_loss=%.6f | final_loss=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], edge_loss.item(), sal_loss.item(), final_loss.item()))

            tmp_path = './tem_see'
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            if step % 20 == 0:
                vutils.save_image(torch.sigmoid(out_final[0,:,:,:].data), tmp_path + '/iter%d-sal-0.jpg' % step, normalize=True, padding=0)
                vutils.save_image(image[0,:,:,:].data, tmp_path + '/iter%d-sal-data.jpg' % step, padding=0)
                vutils.save_image(mask[0,:,:,:].data, tmp_path + '/iter%d-sal-target.jpg' % step, padding=0)

        if epoch > cfg.epoch*1/2:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

if __name__=='__main__':
    train(dataset, CRRN)