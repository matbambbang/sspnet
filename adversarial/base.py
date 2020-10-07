import torch
import torch.nn as nn
from torch.autograd import Variable


class AttackBase(object) :
    def __init__(self, model=None, norm=False, discrete=True, device=None) :
        self.model = model
        self.norm = norm
        # Normalization are needed for RGB images.
        if self.norm == "cifar10" :
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        if self.norm == "normal" or self.norm == "tiny" :
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
            print("mean : ", self.mean)
            print("std  : ", self.std)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None) :
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self) :
        raise NotImplementedError

    def normalize(self,x) :
        if self.norm != "mnist" and self.norm != "fmnist" :
            y = x.clone().to(x.device)
            y[:,0,:,:] = (y[:,0,:,:] - self.mean[0]) / self.std[0]
            y[:,1,:,:] = (y[:,1,:,:] - self.mean[1]) / self.std[1]
            y[:,2,:,:] = (y[:,2,:,:] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self,x) :
        if self.norm != "mnist" and self.norm != "fmnist" :
            y = x.clone().to(x.device)
            y[:,0,:,:] = y[:,0,:,:] * self.std[0] + self.mean[0]
            y[:,1,:,:] = y[:,1,:,:] * self.std[1] + self.mean[1]
            y[:,2,:,:] = y[:,2,:,:] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x) :
        return torch.round(x * 255) / 255

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False) :
        if not inverse_normalized :
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf" :
            clamp_delta = torch.clamp(x_adv-x_nat, -bound, bound)
        else :
            #NOTE: This case only considers p=2. When p!=2, the other projection should be needed.
            clamp_delta = x_adv-x_nat
            for batch_index in range(clamp_delta.size(0)) :
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)

                if image_norm > bound :
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)
