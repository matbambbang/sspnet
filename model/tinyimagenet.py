import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, Bottleneck, ResBlock, SSPBlock2, SSPBlock3, RKBlock2
from .block import ArkBlock
# from .block import FlexResBlock, EResBlock

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


class TinyImagenetModule(nn.Module) :
    def __init__(self, block, layers=1, num_classes=200, init_channel=16, norm_type="b", downsample_type="r") :
        super(TinyImagenetModule,self).__init__()
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
        self.sub3 = self._subsample(channel, channel * 2, norm_type=norm_type, block_type=downsample_type)
        channel *= 2
        self.block4 = nn.Sequential(
            *[block(channel, channel, norm_type=norm_type) for _ in range(layers)]
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

    def forward(self, x) :
        out = self.conv(x)
        out = self.block1(out)
        out = self.sub1(out)

        out = self.block2(out)
        out = self.sub2(out)

        out = self.block3(out)
        out = self.sub3(out)

        out = self.block4(out)

        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()

class TinyImagenetModule_ARK(nn.Module) :
    def __init__(self, block, layers=1, num_classes=200, init_channel=16, norm_type="b", downsample_type="r", group1=ResBlock, group2=ArkBlock, group3=SSPBlock2, group4=SSPBlock2, a21=0.25, b10=1.0, a_logic=False, b_logic=True) :
        super(TinyImagenetModule_ARK,self).__init__()
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
        self.sub3 = self._subsample(channel, channel*2, norm_type=norm_type,block_type=downsample_type)
        channel *= 2
        self.block4 = nn.Sequential(
            *[group4(channel, channel, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic) for _ in range(layers)]
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
        out = self.sub3(out)

        out = self.block4(out)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def loss(self) :
        return nn.CrossEntropyLoss()

def tinyimagenet_model(block_type="res", layers=6, norm_type="b", architecture="imagenet") :
    if architecture == "imagenet" :
        if block_type == "res" :
            return TinyImagenetModule(block=ResBlock, layers=layers, norm_type=norm_type)
        elif block_type == "ssp2" :
            return TinyImagenetModule(block=SSPBlock2, layers=layers, norm_type=norm_type)
        elif block_type == "ssp3" :
            return TinyImagenetModule(block=SSPBlock3, layers=layers, norm_type=norm_type)
        elif block_type == "midrk2" :
            return TinyImagenetModule(block=RKBlock2, layers=layers, norm_type=norm_type)
        elif block_type == "ark" :
            block1 = ArkBlock
            block2 = ResBlock
            block3 = SSPBlock2
            block4 = SSPBlock2
            a21=0.25
            b10=1.0
            a_logic=False
            b_logic=True
            return TinyImagenetModule_ARK(block=block_type, layers=layers, norm_type=norm_type, group1=block1, group2=block2, group3=block3, group4=block4, a21=0.25, b10=1.0, a_logic=a_logic, b_logic=b_logic)

def tinyimagenet_ark_model(block_type="res", layers=6, norm_type="b", block1=None, block2=None, block3=None, block4=None, a21=0.25, b10=1.0, a_logic=False, b_logic=True) :

    if block_type == "res" :
        return TinyImagenetModule(block=ResBlock_Bottleneck, layers=layers, norm_type=norm_type)
    elif block_type == "ssp2" :
        return TinyImagenetModule(block=SSPBlock2_Bottleneck, layers=layers, norm_type=norm_type)
    elif block_type == "ssp3" :
        return TinyImagenetModule(block=SSPBlock3_Bottleneck, layers=layers, norm_type=norm_type)
    elif block_type == "ark" or block_type == 'multi':
        block_type = ArkBlock_Bottleneck
        print(f"Start: Block1- {block1}, Block2- {block2}, Block3- {block3}, Alpha-{a21}-{a_logic}, Beta-{b10}-{b_logic}")
        # Block 1
        if block1 == "ssp2":
            block1 = SSPBlock2_Bottleneck
        elif block1 == "ssp3":
            block1 = SSPBlock3_Bottleneck
        elif block1 == "res":
            block1 = ResBlock_Bottleneck
        elif block1 == "ark":
            block1 = ArkBlock_Bottleneck
        # Block 2
        if block2 == "ssp2":
            block2 = SSPBlock2_Bottleneck
        elif block2 == "ssp3":
            block2 = SSPBlock3_Bottleneck
        elif block2 == "res":
            block2 = ResBlock_Bottleneck
        elif block2 == "ark":
            block2 = ArkBlock_Bottleneck
        # Block 3
        if block3 == "ssp2":
            block3 = SSPBlock2_Bottleneck
        elif block3 == "ssp3":
            block3 = SSPBlock3_Bottleneck
        elif block3 == "res":
            block3 = ResBlock_Bottleneck
        elif block3 == "ark":
            block3 = ArkBlock_Bottleneck
        # Block 4
        if block4 == "ssp2":
            block4 = SSPBlock2_Bottleneck
        elif block4 == "ssp3":
            block4 = SSPBlock3_Bottleneck
        elif block4 == "res":
            block4 = ResBlock_Bottleneck
        elif block4 == "ark":
            block4 = ArkBlock_Bottleneck
        if a_logic == 'False':
            a_logic = False
        else:
            a_logic = True
        if b_logic == 'False':
            b_logic = False
        else:
            b_logic = True
        print(f"End: Block1- {block1}, Block2- {block2}, Block3- {block3}, Block4- {block4}, Alpha- {a21}-{a_logic}, Beta- {b10}-{b_logic}")
        return TinyImagenetModule_ARK(block=block_type, layers=layers, norm_type=norm_type, group1=block1, group2=block2, group3=block3, group4=block4, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic)
