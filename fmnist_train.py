import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from dataloader import get_fmnist_loaders
from utils import init_logger, RunningAverageMeter, accuracy
from model.mnist import mnist_model
from container import trainer


parser = argparse.ArgumentParser("FMNIST")
parser.add_argument("--model", type=str, default="res", choices=["res", "ssp2", "ssp3", "ark"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--block", type=eval, default=3)
parser.add_argument("--hist", type=eval, default=False)
parser.add_argument("--norm", type=str, default="g")
parser.add_argument("--save", type=str, default="exp")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--adv", type=str, default="none", choices=["none", "fgsm", "pgd", "ball"])
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--opt", type=str, default="adam", choices=["sgd", "adam", "rms"])
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--init", type=str, default="kn")

args = parser.parse_args()


if __name__ == "__main__" :
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if os.path.exists(args.save) :
        raise NameError("previous experiment '{}' already exists!".format(args.save))
    os.makedirs(args.save)

    logger = init_logger(logpath=args.save, experiment_name="logs-"+args.model)
    logger.info(args)

    if args.gpu >= 0 :
       args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else :
        args.device = torch.device("cpu")
    model = mnist_model(args.model, layers=args.block, norm_type=args.norm, init_option=args.init)
    logger.info(model)
    model.to(args.device)

    train_loader, test_loader, train_eval_loader = get_fmnist_loaders(batch_size=64, test_batch_size=500)
    loader = {"train_loader": train_loader, "train_eval_loader": train_eval_loader, "test_loader": test_loader}
    if args.opt == "sgd" :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,140], gamma=0.1)
    elif args.opt == "adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif args.opt == "rms" :
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        scheduler = None
    
    adv_train = args.adv if args.adv != "none" else None
    model = trainer(model, logger, loader, args, "fmnist", optimizer, scheduler, adv_train=adv_train)
