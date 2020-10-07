import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reference: https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
def conv3x3(in_planes, out_planes, stride=1, bias=True) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(planes, norm_type="b") :
    if norm_type == "g" :
        return nn.GroupNorm(num_groups=min(32,planes), num_channels=planes)
    elif norm_type == "l" :
        return nn.GroupNorm(num_groups=1, num_channels=planes)
    elif norm_type == "i" :
        return nn.GroupNorm(num_groups=planes, num_channels=planes)
    return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module) :
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, coef=1, norm_type="b") :
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.n1 = norm(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.n2 = norm(planes, norm_type)
        self.downsample = downsample
        self.coef = coef
        self.residual = nn.Sequential(
                self.conv1,
                self.n1,
                self.relu,
                self.conv2,
                self.n2
                )

    def forward(self, x) :
        identity = x 

        out = self.residual(x)
        if self.downsample is not None :
            identity = self.downsample(x)
        out += self.coef * identity
        return self.relu(out)


class Bottleneck(nn.Module) :
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None) :
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(inplanes, plnaes)
        self.n1 = norm(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.n2 = norm(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.n3 = norm(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) :
        identity = x
        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.n3(out)

        if self.downsample is not None :
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def initialize(self) :
        return None


class ConvBlock(nn.Module) :
    expansion = 1

    def __init__(self, inplanes, planes, coef=1, stride=1, norm_type="b") :
        super(ConvBlock,self).__init__()
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)

        self.convs = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )

    def forward(self, x) :
        return self.convs(x)


class ResBlock(nn.Module) :
    # This ResBlock is build based on https://arxiv.org/abs/1603.05027
    expansion = 1

    def __init__(self, inplanes, planes, coef=1, stride=1, downsample=None, norm_type="b", **kwargs) :
        super(ResBlock,self).__init__()
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.coef = coef

        self.residual = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )


    def forward(self, x) :
        shortcut = x

        if self.downsample is not None :
            shortcut = self.relu(self.n1(x))
            shortcut = self.downsample(shortcut)

        out = self.coef * self.residual(x)

        return out + shortcut

    def initialize(self) :
        nn.init.zeros_(self.n1.weight)


class ResBlock_Bottleneck(nn.Module) :
    # This ResBlock is build based on https://arxiv.org/abs/1603.05027
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type="b") :
        super(ResBlock_Bottleneck,self).__init__()
        self.conv1 = conv1x1(inplanes,planes)
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.n3 = norm(planes, norm_type)

    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.n1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.n3(out)

        if self.downsample is not None :
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def initialize(self) :
        return None


class RKBlock2(nn.Module) :
    # mid-RK2 block. Please see equation (11) in Section 3.2.
    expansion = 1

    def __init__(self, inplanes, planes, coef=1, stride=1, downsample=None, norm_type="b") :
        super(RKBlock2,self).__init__()
        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.coef = coef

        self.residual = nn.Sequential(
                self.n1,
                self.relu,
                self.conv1,
                self.n2,
                self.relu,
                self.conv2
                )


    def forward(self, x) :
        shortcut = x

        out = x + 0.5 * self.residual(x)
        out = out + self.residual(out)

        return out


class SSPBlock2(nn.Module) :
    # SSP2-block. Please see equation (9) in Section 3.2.
    expansion = 1

    def __init__(self, inplanes, planes, block=ResBlock, coef=1, stride=1, norm_type="b", **kwargs) :
        super(SSPBlock2,self).__init__()
        self.block = block(inplanes, planes, coef=coef, norm_type=norm_type)

    def forward(self, x) :
        x1 = self.block(x)
        out = 0.5*x + 0.5*self.block(x1)
        return out

    def initialize(self) :
        self.block.initialize()


class SSPBlock3(nn.Module) :
    # SSP3-block. Please see equation (10) in Section 3.2.
    expansion = 1
    
    def __init__(self, inplanes, planes, block=ResBlock, coef=1, stride=1, norm_type="b", **kwargs) :
        super(SSPBlock3,self).__init__()
        self.block = block(inplanes, planes, coef=coef, norm_type=norm_type)

    def forward(self, x) :
        x1 = self.block(x)
        x2 = 0.75*x + 0.25*self.block(x1)
        out = 1/3 * x + 2/3 * self.block(x2)
        return out

    def initialize(self) :
        self.block.initialize()


class ArkBlock(nn.Module) :
    # Adaptive Runge-Kutta block. Please see equation (13)~(15) in Section 3.2.
    expansion = 1

    def __init__(self, inplanes, planes, block=ResBlock, coef=1, stride=1, norm_type="b", a21=-1.0, b10=0.5, a_logic=False, b_logic=True) :
        super(ArkBlock, self).__init__()
        if a_logic == 'False':
            a_logic = False
        else:
            a_logic = True
        if b_logic == 'False':
            b_logic = False
        else:
            b_logic = True

        self.a21 = nn.Parameter(torch.tensor(a21), requires_grad=a_logic)
        self.b10 = nn.Parameter(torch.tensor(b10), requires_grad=b_logic)

        self.n1 = norm(inplanes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.n2 = norm(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.coef = coef

        self.f = nn.Sequential(
            self.n1,
            self.relu,
            self.conv1,
            self.n2,
            self.relu,
            self.conv2
        )

    def forward(self, x) :
        m = nn.Sigmoid()
        self.a20 = 1 - m(self.a21)
        self.b20 = 1 - 1/(2*m(self.b10)) - m(self.a21)*m(self.b10)
        self.b21 = 1/(2*m(self.b10))
        x1 = x + m(self.b10)*self.f(x)
        out = self.a20*x+self.b20*self.f(x)+m(self.a21)*x1+self.b21*self.f(x1)
        return out

    def initialize(self) :
        self.block.initialize()

    def coef_visualize(self) :
        print("a20 : {}(recommended value: 1)".format(self.a20))
        print("a21 : {}(recommended value: 1)".format(self.a21))
        print("b10 : {}(recommended value: 1)".format(self.b10))
        print("b20 : {}(recommended value: 1)".format(self.b20))
        print("b21 : {}(recommended value: 1)".format(self.b21))


def coef_controller(module, value=1.) :
    module.block.coef = value
