import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_cifar10_loaders, inf_generator
from container import test
from utils import accuracy, init_logger
from model.block import ResBlock, SSPBlock2, SSPBlock3, RKBlock2, ArkBlock
from model.cifar10 import cifar_model, PGModule, PGModule_ARK
from adversarial import FGSM, LinfPGD

parser = argparse.ArgumentParser("Lipschitzness evaluation")
parser.add_argument("--model", type=str, default="res", choices=["res", "ssp2", "ssp3", "midrk2", "ark"])
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--block", type=int, default=6)

#parser.add_argument("--block1", type=str, default="ssp2", choices=["res", "ssp2", "ssp3", "ark"])
#parser.add_argument("--block2", type=str, default="ark", choices=["res", "ssp2", "ssp3", "ark"])
#parser.add_argument("--block3", type=str, default="res", choices=["res", "ssp2", "ssp3", "ark"])
#parser.add_argument("--block4", type=str, default="res", choices=["res", "ssp2", "ssp3", "ark"])

parser.add_argument("--norm_type", type=int, default=1)
parser.add_argument("--attack", type=str, default="pgd")
parser.add_argument("--eps", type=float, default=8.)
parser.add_argument("--alpha", type=float, default=2.)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--bsize", type=int, default=100)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

def eval(model, loader, device, adv=None, index=None) :
    model.eval()
    total_correct = 0
    total_loss = []
    criterion = nn.CrossEntropyLoss().to(device)

    for i, (x,y) in enumerate(loader) :
        if adv is not None :
            x = attack.perturb(x.to(device),y.to(device),device=device)
        x = x.to(device)
        if index is not None :
            pred = model(x,index)
        else :
            pred = model(x)
        pred_class = torch.argmax(pred.cpu().detach(), dim=-1)
        correct = (pred_class == y.cpu())

        # loss
        y = y.to(device)
        loss = criterion(pred,y).cpu().detach().numpy()

        total_loss.append(loss)
        total_correct += torch.sum(correct).item()
    return total_correct/len(loader.dataset), np.mean(total_loss)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:"+str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
logger = init_logger(logpath=args.load, experiment_name="check_lipschitzness")

_, loader, _ = get_cifar10_loaders(data_aug=True, test_batch_size=args.bsize)

model = cifar_model(args.model, layers=args.block, norm_type="b")
model.load_state_dict(torch.load(os.path.join(args.load,"model_acc.pt"), map_location="cpu")['state_dict'], strict=False)
if args.model == "res" :
    model_part = PGModule(block=ResBlock, layers=args.block)
elif args.model == "ssp2" :
    model_part = PGModule(block=SSPBlock2, layers=args.block)
elif args.model == "ssp3" :
    model_part = PGModule(block=SSPBlock3, layers=args.block)
elif args.model == "midrk2" :
    model_part = PGModule(block=RKBlock2, layers=args.block)
elif args.model == "ark" :
    # SSP-adap
    group1 = ArkBlock
    group2 = ResBlock
    group3 = SSPBlock2
    model_part = PGModule_ARK(group1=group1, group2=group2, group3=group3, layers=args.block)
model_part.load_state_dict(torch.load(os.path.join(args.load,"model_acc.pt"), map_location="cpu")['state_dict'], strict=False)
model.eval()
model_part.eval()
model.to(device)
model_part.to(device)
criterion = nn.CrossEntropyLoss().to(device)

logger.info("="*80)

args.eps /= 255
args.alpha /= 255
adv = LinfPGD(model, bound=args.eps, step=args.alpha, iters=args.iters, random_start=False, norm="cifar10", device=device)
#adv = EpsilonAdversary(model, epsilon=args.eps, repeat=10, dist="uniform", norm="cifar10", device=device)

diff_arr = [[], [], []]

for i, (x,y) in enumerate(loader) :
    x = x.to(device)
    y = y.to(device)
    x_adv = adv.perturb(x,y)

    partial_diff = []
    for j in range(3) :
        with torch.no_grad() :
            before_nat, after_nat = model_part(x,output_group=j)
            before_adv, after_adv = model_part(x_adv,output_group=j)
            if args.norm_type == 2 :
                diff_ratio = torch.norm(after_adv - after_nat, dim=1) / torch.norm(before_adv-before_nat, dim=1)
            elif args.norm_type == 1 :
                diff_ratio = torch.norm(after_adv - after_nat, p=1, dim=1) / torch.norm(before_adv-before_nat, p=1, dim=1)
            diff_arr[j].append(diff_ratio)

# diff_mean = torch.tensor(diff_arr).mean(dim=0)
diff_ratio_mean = [torch.stack(elem).mean() for elem in diff_arr]

for idx in range(3) :
    logger.info("Partial {} | Diff : {:.5f}".format(idx, diff_ratio_mean[idx]))
