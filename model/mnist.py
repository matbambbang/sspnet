import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import conv3x3, conv1x1, norm
from .block import ConvBlock, ResBlock, SSPBlock2, SSPBlock3, ArkBlock
from .utils import Flatten


class MNISTModule(nn.Module) :
    def __init__(self,
            block,
            layers=1,
            channels=64,
            stride=1,
            coef=1,
            classes=10,
            norm_type="b",
            init_option="basic") :
        super(MNISTModule,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(1,channels,3,1), norm(channels, norm_type), self.relu, nn.Conv2d(channels,channels,4,2,1),
                norm(channels, norm_type), self.relu, nn.Conv2d(channels,channels,4,2,1)
                )
        self.blocks = nn.Sequential(
                *[block(inplanes=channels, planes=channels, stride=stride, coef=coef, norm_type=norm_type) for _ in range(layers)]
                )
        self.linear = nn.Sequential(
                norm(channels, norm_type), self.relu, nn.AdaptiveAvgPool2d((1,1)), Flatten(), nn.Linear(channels,10)
                )

        if init_option == "alter" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight / torch.sqrt(torch.tensor(2.)))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
#                            print(sm.weight.abs().sum())
                            sm.weight = nn.Parameter(sm.weight / torch.sqrt(torch.tensor(3.)))
#                            print(sm.weight.abs().sum())
        if init_option == "back" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * ((torch.sqrt(torch.tensor(5.)) - 1) / 2))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * 0.75487767)

        elif init_option == "alter2" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.sqrt(torch.tensor(3.))-1))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.tensor(1.26479665)))

        elif init_option == "alter3" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.sqrt(torch.sqrt(torch.tensor(2.))-1)))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.sqrt(torch.tensor(1.26479665))))

        elif init_option == "upscale" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.tensor(2.)))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.tensor(3.)))

        elif init_option == "ortho" :
            for m in self.modules() :
                if isinstance(m, nn.Conv2d) :
                    nn.init.orthogonal_(m.weight)
 
        elif init_option == "kn" :
            for m in self.modules() :
                if isinstance(m, nn.Conv2d) :
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) :
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


        if init_option == "back" :
            for m in self.modules() :
                if isinstance(m, SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * ((torch.sqrt(torch.tensor(5.)) - 1) / 2))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * 1.26479665)


    def forward(self,x) :
        out = self.downsample(x)
        out = self.blocks(out)
        out = self.linear(out)
        return out

    def loss(self) :
        return nn.CrossEntropyLoss()

class MiddleCut(MNISTModule) :
    def __init__(self, block, layers=1, channels=64, stride=1, coef=1, classes=10, norm_type="b") :
        super(MiddleCut,self).__init__(block=block, layers=layers, channels=channels, stride=stride, coef=coef, classes=classes, norm_type=norm_type)

    def forward(self,x,index=None) :
        out = self.downsample(x)
        for i in range(index) :
            out = self.blocks[i](out)

        out = nn.AdaptiveAvgPool2d((1,1))(out)
        return out.view(out.size(0),-1)

def mnist_model(block_type="conv", layers=3, norm_type="b", init_option="basic") :
    if block_type == "conv" :
        return MNISTModule(block=ConvBlock, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "res" :
        return MNISTModule(block=ResBlock, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ssp2" :
        return MNISTModule(block=SSPBlock2, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ssp3" :
        return MNISTModule(block=SSPBlock3, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ssp3rk" :
        return MNISTModule(block=SSPBlock3_RK, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ark" :
        return MNISTModule(block=ArkBlock, layers=layers, norm_type=norm_type, init_option=init_option)

## Experiments
#def resnet_shallow(block=ResBlock, layers=3) :
#    return MNISTModule(block, layers=layers, coef=0.5)
#
#def sspnet3_rks(block=SSPBlock3_RK, layers=3) :
#    return MNISTModule(block, layers=layers, coef=0.5)
#
#def adaptive2(block=AdaptiveBlock2, layers=3) :
#    return MNISTModule(block, layers=layers)
#
#def adaptive3(block=AdaptiveBlock3, layers=3) :
#    return MNISTModule(block, layers=layers)
#
#def ssp(block=SSPBlock, layers=3) :
#    return MNISTModule(block, layers=layers)
