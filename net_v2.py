#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from channel_graph import ChannelGCN


############## 初始化 和backbone#################################
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m, nn.Sequential):
            weight_init(m)

        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg      = cfg
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        if self.cfg.snapshot:
            pass
        else:
            self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []

        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        tmp_x.append(out1)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        tmp_x.append(out2)
        out3 = self.layer2(out2)
        tmp_x.append(out3)
        out4 = self.layer3(out3)
        tmp_x.append(out4)
        out5 = self.layer4(out4)
        tmp_x.append(out5)
        return tmp_x

    def initialize(self):
        self.load_state_dict(torch.load('./res/resnet50-19c8e357.pth'), strict=False)
######################  backbone end ###################################################


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(list_k[1][i])))
        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

    def initialize(self):
        weight_init(self)

class MergeLayer(nn.Module):
    def __init__(self,list_k):
        super(MergeLayer, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for j, ik in enumerate(list_k):
            if j > 0:
                trans.append(nn.Sequential(nn.Conv2d(64, 64, ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                           nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64)))
            up.append(nn.Sequential(nn.Conv2d(64, 64, ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.BatchNorm2d(64)))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))
        trans.append(nn.Sequential(nn.Conv2d(64, 64, 7, 1, padding=3), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                   nn.Conv2d(64, 64, 3, 1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64)))

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)

        self.relu = nn.ReLU()
        self.channel_decrease = nn.Sequential(nn.Conv2d(320, 64, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.channel_graph = ChannelGCN(64)

        self.final_score = nn.Sequential(nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 1, 3, 1, padding=1))

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []

        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        sal_feature.append(tmp)
        U_tmp = tmp
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), x_size, mode='bilinear',
                                    align_corners=True))
        for j in range(2, num_f):
            i = num_f - j

            U_tmp = list_x[i] + F.interpolate((self.trans[i](U_tmp)), list_x[i].size()[2:], mode='bilinear',
                                              align_corners=True)
            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            sal_feature.append(tmp)
            if i > 1:
                up_sal.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True))
            if i == 1:
                up_edge.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True))

        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear',
                                          align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)
        up_edge.append(F.interpolate(self.score[0](tmp), x_size, mode='bilinear', align_corners=True))

        edge_feat_size = edge_feature[0].size()[2::]

        mergelayer1 = F.interpolate(sal_feature[0], edge_feat_size, mode='bilinear',
                                    align_corners=True)
        for i, feature in enumerate(sal_feature):
            if i > 0:
                feature = F.interpolate(feature, edge_feat_size, mode='bilinear', align_corners=True)
                mergelayer1 = torch.cat((mergelayer1, feature), dim=1)
        mergelayer1 = torch.cat((mergelayer1, edge_feature[0]), dim=1)
        mergelayer1,res_graph = self.channel_graph(mergelayer1)
        final_score = F.interpolate(self.final_score(mergelayer1), x_size, mode='bilinear', align_corners=True)
        return up_edge, up_sal, final_score

    def initialize(self):
        weight_init(self)

# network
config = {'convert': [[64, 256, 512, 1024, 2048], [64, 64, 64, 64, 64]],
          'merge': [[64, 64, 64, 3, 1, []], [64, 64, 64, 3, 1, [64, 64]],
                    [64, 0, 64, 5, 2, [64, 64]], [64, 0, 64, 5, 2,[64, 64]],
                    [64, 0, 64, 7, 3, [64, 64]]]}

class CRRN(nn.Module):
    def __init__(self, cfg):
        super(CRRN, self).__init__()
        self.cfg = cfg
        self.base = ResNet(cfg)
        self.convert = ConvertLayer(config['convert'])
        self.merge = MergeLayer(config['merge'])
        self.initialize()

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge = self.base(x)
        conv2merge = self.convert(conv2merge)
        up_edge, up_sal, final_score = self.merge(conv2merge, x_size)
        return up_edge, up_sal, final_score

    def initialize(self):
        if self.cfg.snapshot:
            pass
        else:
            weight_init(self)

if __name__ == "__main__":

    net = CRRN().cuda()
    img = torch.rand(3, 3, 256, 256).cuda()
    _,_,output,_ = net(img)
    print(output.shape)