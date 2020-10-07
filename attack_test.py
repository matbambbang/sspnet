import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle

from dataloader import get_mnist_loaders, get_cifar10_loaders, get_fmnist_loaders
from dataloader_tiny_imagenet import get_tinyimagenet_loaders
from utils import init_logger, RunningAverageMeter, accuracy
from container import test
from adversarial import FGSM, LinfPGD
from adversarial import EpsilonAdversary
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("Attack")

parser.add_argument("--model", type=str, default="conv", choices=["res", "ssp2", "ssp3", "ark", "midrk2"])
parser.add_argument("--eval", type=str, default="cifar10", choices=["mnist", "cifar10", "fmnist", "tiny"])
parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "pgd", "ball"])
parser.add_argument("--archi", type=str, default="cifar10", choices=["imagenet", "cifar10"])
parser.add_argument("--multi", type=eval, default=False)
parser.add_argument("--metric", type=str, default="Linf", choices=["Linf"])
parser.add_argument("--block", type=int, default=6)
parser.add_argument("--load", type=str, default="exp")
parser.add_argument("--bsize", type=int, default=100)
parser.add_argument("--norm", type=str, default="b")
parser.add_argument("--eps", type=float, default=8.)
parser.add_argument("--alpha", type=float, default=2.)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bin", type=eval, default=False)
parser.add_argument("--crit", type=str, default="acc")
parser.add_argument("--adv_save", type=eval, default=True)

args = parser.parse_args()

def l1_distance(tensor1, tensor2) :
    assert tensor1.size() == tensor2.size()
    residual = torch.abs(tensor1-tensor2)
    if len(tensor1.size()) == 4 :
        return residual.sum() / (tensor1.size(0)*tensor1.size(1)*tensor1.size(2)*tensor1.size(3))
    return residual.sum()

def l2_distance(tensor1, tensor2) :
    assert tensor1.size() == tensor2.size()
    residual = (tensor1 - tensor2) ** 2
    if len(tensor1.size()) == 4 :
        return torch.sqrt(torch.sum(residual, (1,2,3))).sum() / tensor1.size(0)
    return residual.sum()

def adversarial_attack(model, logger, target_loader, args, **kwargs) :
    logger.info("="*80)
    logger.info("Natural result")
    orig_acc, orig_loss = test(model, target_loader, args.device)
    logger.info("Accuracy : {:.4f}".format(orig_acc))
    logger.info("Loss : {:.4f}".format(orig_loss))
    logger.info("-"*80)
    logger.info("Attack parameters : eps={}".format(args.eps))

    repeat = 1
    if args.attack == "ball" :
        repeat = 10
    alpha = None
    k = None
    norm = args.eval
    if args.metric == "Linf" :
        if args.eval == "cifar10" or args.eval == "tiny" :
            args.eps /= 255
            args.alpha /= 255
        if args.attack == "fgsm" :
            attack_module = FGSM(model, bound=args.eps, norm=norm, device=args.device)
        elif args.attack == "pgd" :
            attack_module = LinfPGD(model, bound=args.eps, step=args.alpha, iters=args.iters, random_start=False, norm=norm, device=args.device)
        elif args.attack == "ball" :
            attack_module = EpsilonAdversary(model, epsilon=args.eps, repeat=repeat, dist="uniform", norm=norm, device=args.device)
    elif args.metric == "L2" :
        # This does not work in this repo.
        assert ValueError("L2 robustness evaluation does not work in this codes.")
        bound = args.eps
        if args.attack in ["bim", "pgd"] :
            k = 5
            alpha = 2 * args.eps / k
        if args.attack == "fgm" :
            attack_module = GradientAttack(model, bound=bound, norm=norm, device=args.device)
   
    # Adversarial attack
    writer=SummaryWriter(log_dir=os.path.join(args.load, args.attack+"_"+args.metric+"_"+str(args.eps)))
    device =args.device
    total_correct = 0
    criterion = nn.CrossEntropyLoss().to(args.device)
    l1_arr = []
    l2_arr = []
    adv_saver = []
    for i, (x,y) in enumerate(target_loader) :
        if attack_module is not None :
            x_nat = x.detach().clone().to(device)
            x = attack_module.perturb(x.to(device), y.to(device), device=device)
            if repeat != 1 :
                y = torch.cat([y for _ in range(repeat)])

            if args.attack != "ball" :
                l1_dist = l1_distance(attack_module.inverse_normalize(x_nat), attack_module.inverse_normalize(x))
                l2_dist = l2_distance(attack_module.inverse_normalize(x_nat), attack_module.inverse_normalize(x))
                writer.add_scalar("L1", l1_dist, i)
                writer.add_scalar("L2", l2_dist, i)
                l1_arr.append(l1_dist.cpu().numpy())
                l2_arr.append(l2_dist.cpu().numpy())

        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y).cpu().detach().numpy()
        predicted_class = torch.argmax(pred.cpu().detach(), dim=1)
        correct = (predicted_class == y.cpu())
        total_correct += torch.sum(correct).item()

        if args.adv_save :
            adv_saver.append((x.cpu(), y.cpu()))

        if args.eval == "cifar10" :
            x_nat = attack_module.inverse_normalize(x_nat)
            x = attack_module.inverse_normalize(x)
        nat_image = torchvision.utils.make_grid(x_nat.cpu(), nrow=5, scale_each=False)
        adv_image = torchvision.utils.make_grid(x.cpu(), nrow=5, scale_each=False)
        writer.add_image("natural_image", nat_image, i)
        writer.add_image("adversarial_image", adv_image, i)

    adv_acc = total_correct / (len(target_loader.dataset) * repeat)
    writer.add_text("natural_acc", str(orig_acc), 1)
    writer.add_text("natural_loss", str(orig_loss), 1)
    writer.add_text("adversarial_acc", str(adv_acc), 1)
    if args.attack != "ball" :
        writer.add_text("L1_distance", str(np.mean(l1_arr)), 1)
        writer.add_text("L2_distance", str(np.mean(l2_arr)), 1)
    if alpha is not None :
        writer.add_text("alpha(stepsize)", str(alpha), 1)
    if k is not None :
        writer.add_text("Iteration", str(k), 1)

    writer.close()
    if args.adv_save :
        if not os.path.exists(os.path.join(args.load, args.attack+"_"+str(args.eps))) :
            os.makedirs(os.path.join(args.load, args.attack+"_"+str(args.eps)))
        with open(os.path.join(args.load, args.attack+"_"+str(args.eps), "adversary.pkl"), "wb") as f :
            pickle.dump(adv_saver,f)

    logger.info("Attacked Accuracy : {:.4f}".format(adv_acc))
    logger.info("Finished")
    logger.info("="*80)
 
if __name__ == "__main__" :
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.attack == "ortho" :
        logger = init_logger(logpath=args.load, experiment_name="attack-"+str(args.attack)+"-"+str(args.eps)+"-"+str(args.step))
    else :
        logger = init_logger(logpath=args.load, experiment_name="attack-"+str(args.attack)+"-"+str(args.eps))
    
    if args.eval == "mnist" or args.eval == "fmnist" :
        in_channels = 1
    else :
        in_channels = 3

    if args.eval == "mnist" or args.eval == "norm" or args.eval == "fmnist" :
        from model.mnist import mnist_model
        model = mnist_model(args.model, layers=args.block, norm_type=args.norm)
    elif args.eval == "cifar10" :
        from model.cifar10 import cifar_model
        model = cifar_model(args.model, layers=args.block, norm_type=args.norm)
    elif args.eval == "tiny" :
        from model.tinyimagenet import tinyimagenet_model
        model = tinyimagenet_model(args.model, layers=args.block, norm_type=args.norm, architecture=args.archi)
    logger.info(model)
    if args.multi :
        model = nn.DataParallel(model).cuda()

    if args.crit == "acc" :
        model_dict = torch.load(os.path.join(args.load,"model_acc.pt"), map_location=str(args.device))
    elif args.crit == "last" :
        model_dict = torch.load(os.path.join(args.load,"model_final.pt"), map_location=str(args.device))
    elif args.crit == "loss" :
        model_dict = torch.load(os.path.join(args.load, "model_loss.pt"), map_location=str(args.device))

    # When you does not using Adaptive Runge-Kutta block, you could use strict=True for all the other models.
    # model.load_state_dict(model_dict["state_dict"], strict=True)
    model.load_state_dict(model_dict["state_dict"], strict=False)
    if args.multi :
        model.cuda()
    else :
        model.to(args.device)

    if args.eval == "mnist" :
        train_loader, test_loader, train_eval_loader = get_mnist_loaders(data_aug=False, test_batch_size=args.bsize)
    elif args.eval == "cifar10" :
        train_loader, test_loader, train_eval_loader = get_cifar10_loaders(data_aug=True, test_batch_size=args.bsize)
    elif args.eval == "norm" :
        train_loader, test_loader, train_eval_loader = norm_mnist_loaders()
    elif args.eval == "fmnist" :
        train_loader, test_loader, train_eval_loader = get_fmnist_loaders(test_batch_size=args.bsize)
    elif args.eval == "tiny" :
        train_loader, test_loader, train_eval_loader = get_tinyimagenet_loaders(data_aug=False, test_batch_size=args.bsize)

    model.eval()
    adversarial_attack(model, logger, test_loader, args)
