import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, Bottleneck, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ConvBlock, ArkBlock

# Reference: https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
def make_block_sequence(block, inplanes=64, planes=64, blocks=2, stride=1) :
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes*block.expansion, stride),
                norm(planes*block.expansion)
                )

    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample))
    nextplanes = planes * block.expansion
    for _ in range(1,blocks) :
        layers.append(block(nextplanes, planes))

    return nn.Sequential(*layers), nextplanes


class CIFAR10Module(nn.Module) :
    def __init__(self, block, layers=1, num_classes=10, init_channel=16, norm_type="b", downsample_type="r", init_option="basic", **kwargs) :
        super(CIFAR10Module,self).__init__()
        channel = init_channel
        self.conv = conv3x3(3,channel)
        self.block1 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub1 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block2 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub2 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block3 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)*layers)
                if m.bias is not None : 
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.BatchNorm2d) :
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if init_option == "upscale" :
            print("-"*80)
            for m in self.modules() :
                if isinstance(m ,SSPBlock2) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.tensor(2.)))
                elif isinstance(m, SSPBlock3) :
                    for sm in m.modules() :
                        if isinstance(sm, nn.Conv2d) :
                            sm.weight = nn.Parameter(sm.weight * torch.sqrt(torch.tensor(3.)))

    def _subsample(self, inplanes, planes, stride=2, norm_type="b", block_type="r") :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm(planes, norm_type=norm_type)
                )
        # only supports the ResBlock
        if block_type == "r" :
            return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)
        return BasicBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)

    def forward(self, x) :
        out = self.conv(x)
        out = self.block1(out)
        out = self.sub1(out)

        out = self.block2(out)
        out = self.sub2(out)

        out = self.block3(out)

        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()


class CIFAR10Module_ARK(nn.Module) :
    def __init__(self, block, layers=1, num_classes=10, init_channel=16, norm_type="b", downsample_type="r", group1=ArkBlock, group2=ResBlock, group3=SSPBlock2, a21=0.25, b10=1.0, a_logic=False, b_logic=True) :
        super(CIFAR10Module_ARK,self).__init__()
        channel = init_channel
        self.conv = conv3x3(3,channel)
        self.block1 = nn.Sequential(
                *[group1(channel,channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
                )
        self.sub1 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block2 = nn.Sequential(
                *[group2(channel,channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
                )
        self.sub2 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block3 = nn.Sequential(
                *[group3(channel,channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
                )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)*layers)
                if m.bias is not None :
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.BatchNorm2d) :
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _subsample(self, inplanes, planes, stride=2, norm_type="b", block_type="r", a21=0.25, b10=1.0, a_logic=False, b_logic=True) :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm(planes, norm_type=norm_type)
                )
        # only supports the ResBlock
        if block_type == "r" :
            return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)
        return BasicBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic)

    def forward(self, x) :
        out = self.conv(x)
        out = self.block1(out)
        out = self.sub1(out)
        out = self.block2(out)
        out = self.sub2(out)
        out = self.block3(out)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def loss(self) :
        return nn.CrossEntropyLoss()


class PGModule(nn.Module) :
    def __init__(self, block, layers=1, num_classes=10, init_channel=16, output_group=3, norm_type="b", downsample_type="r") :
        super(PGModule,self).__init__()
        channel = init_channel
        self.conv = conv3x3(3,channel)
        self.block1 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub1 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block2 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )
        self.sub2 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block3 = nn.Sequential(
                *[block(channel,channel, norm_type=norm_type) for _ in range(layers)]
                )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)*layers)
                if m.bias is not None :
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.BatchNorm2d) :
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _subsample(self, inplanes, planes, stride=2, norm_type="b", block_type="r") :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm(planes, norm_type=norm_type)
                )
        # only supports the ResBlock
        if block_type == "r" :
            return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)
        return BasicBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)

    def forward(self, x, output_group=0) :
        prev_out = self.conv(x)
        if output_group == 0 :
            out = self.block1(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        prev_out = self.block1(prev_out)
        prev_out = self.sub1(prev_out)
        if output_group == 1 :
            out = self.block2(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        prev_out = self.block2(prev_out)
        prev_out = self.sub2(prev_out)
        if output_group == 2 :
            out = self.block3(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        out = self.block3(prev_out)
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()

class PGModule_ARK(nn.Module) :
    def __init__(self, layers=1, num_classes=10, init_channel=16, norm_type="b", downsample_type="r",
                 group1=ResBlock, group2=ArkBlock, group3=SSPBlock2, a21=0.25, b10=1.0, a_logic=False,
                 b_logic=True):
        super(PGModule_ARK,self).__init__()
        channel = init_channel
        self.conv = conv3x3(3,channel)
        self.block1 = nn.Sequential(
            *[group1(channel, channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
        )
        self.sub1 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block2 = nn.Sequential(
                *[group2(channel, channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
                )
        self.sub2 = self._subsample(channel, channel*2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block3 = nn.Sequential(
                *[group3(channel, channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
                )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channel,num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)*layers)
                if m.bias is not None :
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.BatchNorm2d) :
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _subsample(self, inplanes, planes, stride=2, norm_type="b", block_type="r") :
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm(planes, norm_type=norm_type)
                )
        # only supports the ResBlock
        if block_type == "r" :
            return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)
        return BasicBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)

    def forward(self, x, output_group=0) :
        prev_out = self.conv(x)
        if output_group == 0 :
            out = self.block1(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        prev_out = self.block1(prev_out)
        prev_out = self.sub1(prev_out)
        if output_group == 1 :
            out = self.block2(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        prev_out = self.block2(prev_out)
        prev_out = self.sub2(prev_out)
        if output_group == 2 :
            out = self.block3(prev_out)
            prev_out = prev_out.view(prev_out.size(0),-1)
            out = out.view(out.size(0),-1)
            return prev_out, out

        out = self.block3(prev_out)
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()


class MiddleCut(CIFAR10Module) :
    def __init__(self, block, layers=1, num_classes=10, init_channel=16, norm_type="b", downsample_type="r") :
        super(MiddleCut,self).__init__(block=block,layers=layers,num_classes=num_classes,init_channel=init_channel,norm_type=norm_type,downsample_type=downsample_type)
        self.layers = layers
    
    def forward(self, x, index=None) :
        out = self.conv(x)
        out = self.block1(out)
        out = self.sub1(out)

        out = self.block2(out)
        out = self.sub2(out)

        for i in range(index) :
            out = self.block3[i](out)

        out = self.avg(out)
        out = out.view(out.size(0),-1)
#        return self.fc(out)
        return out
#        if index == self.layers :
#            out = self.fc(out)
#        return out



def cifar_model(block_type="res", layers=6, norm_type="b", init_option="basic") :
    if block_type == "res" :
        return CIFAR10Module(block=ResBlock, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "midrk2" :
        return CIFAR10Module(block=RKBlock2, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ssp2" :
        return CIFAR10Module(block=SSPBlock2, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ssp3" :
        return CIFAR10Module(block=SSPBlock3, layers=layers, norm_type=norm_type, init_option=init_option)
    elif block_type == "ark" :
        # If you want to construct the other combination of blocks, you should use 'cifar_model_ark' function below.
        block1 = ArkBlock
        block2 = ResBlock
        block3 = SSPBlock2
        a21=0.25
        b10=1.0
        a_logic=False
        b_logic=True
        return CIFAR10Module_ARK(block=block_type, layers=layers, norm_type=norm_type, group1=block1, group2=block2, group3=block3, a21=a21, b10=b10,
                a_logic=a_logic, b_logic=b_logic)
 
        
def cifar_model_ark(block_type="res", layers=8, norm_type="b", block1=SSPBlock2, block2=ArkBlock, block3=ArkBlock, a21=0.25, b10=1.0, a_logic=False, b_logic=True) :

    if block_type == "res" :
        return CIFAR10Module_ARK(block=ResBlock, layers=layers, norm_type=norm_type)
    elif block_type == "ssp2" :
        return CIFAR10Module_ARK(block=SSPBlock2, layers=layers, norm_type=norm_type)
    elif block_type == "ssp3" :
        return CIFAR10Module_ARK(block=SSPBlock3, layers=layers, norm_type=norm_type)
    elif block_type == "ark" or block_type == 'multi':
        block_type = ArkBlock
        print(f"Start: Block1- {block1}, Block2- {block2}, Block3- {block3}, Alpha-{a21}-{a_logic}, Beta-{b10}-{b_logic}")
        # Block 1
        if block1 == "ssp2":
            block1 = SSPBlock2
        elif block1 == "ssp3":
            block1 = SSPBlock3
        elif block1 == "res":
            block1 = ResBlock
        elif block1 == "ark":
            block1 = ArkBlock
        # Block 2
        if block2 == "ssp2":
            block2 = SSPBlock2
        elif block2 == "ssp3":
            block2 = SSPBlock3
        elif block2 == "res":
            block2 = ResBlock
        elif block2 == "ark":
            block2 = ArkBlock
        # Block 3
        if block3 == "ssp2":
            block3 = SSPBlock2
        elif block3 == "ssp3":
            block3 = SSPBlock3
        elif block3 == "res":
            block3 = ResBlock
        elif block3 == "ark":
            block3 = ArkBlock
        if a_logic == 'False':
            a_logic = False
        else:
            a_logic = True
        if b_logic == 'False':
            b_logic = False
        else:
            b_logic = True
        print(f"End: Block1- {block1}, Block2- {block2}, Block3- {block3}, Alpha- {a21}-{a_logic}, Beta- {b10}-{b_logic}")
        return CIFAR10Module_ARK(block=block_type, layers=layers, norm_type=norm_type, group1=block1, group2=block2, group3=block3, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic)

