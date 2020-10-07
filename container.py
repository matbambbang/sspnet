import os
import argparse
import time
import pickle
import torch
import torch.nn as nn
import torchvision
import numpy as np
from dataloader import inf_generator
from utils import init_logger, RunningAverageMeter, accuracy, one_hot
from adversarial import AttackBase, FGSM, LinfPGD, EpsilonAdversary
import tqdm

def adv_train_module(attack, model, data_type, iters, device) :
    """
    Adversarial example generator used in adversarial training.
    Please refer Section 4.3 - Adversarial Training.
    For all experiments, we used l_inf norm (p=inf)

    1. MNIST
    eps = 0.3, alpha = 0.05
    2. CIFAR10
    eps = 8/255, alpha=2/255
    3. FMNIST (see supplement)
    eps = 0.1, alpha=0.02
    4. TinyImagenet (see supplement)
    eps = 4/255, alpha=1/255
    """

    # Preprocessing - RGB images should be properly normalized before forward propagation
    if data_type == "cifar10" :
        norm = "cifar"
    elif data_type == "tiny" :
        norm = "tiny"
    else :
        norm = False

    if data_type == "mnist" :
        bound = 0.3
        step = 0.05
    elif data_type == "fmnist" :
        bound = 0.1
        step = 0.02
    elif data_type == "cifar10" :
        bound = 8/255
        step = 2/255
    elif data_type == "tiny" :
        bound = 4/255
        step = 1/255
    
    # When performing adversarial training, the PGD adversary could be start with or without adding random perturbation.
    # If you set the adversary as "pgd", the adversary add random perturbation in the beggining of searching adversarial examples.
    # If you set the adversary as "bim", the adversary does not add random perturbation in the beggining of searching adversarial examples.
    random_start=True if attack == "pgd" else False
    # When you select "ball", samples are augmented by adding random noise drawn from a uniform distribution Uniform(-eps,eps),
    # Each image makes 'repeat_num' augmented images when you choose the random perturbation augmentation. In MNIST case, each image makes 5 augmented images.
    repeat_num = 5 if attack == "ball" else 1
    stats = (repeat_num,)

    if attack == None :
        adv = AttackBase(norm=norm, device=device)
    elif attack == "pgd" or attack == "bim" :
        adv = LinfPGD(model, bound=bound, step=step, iters=iters, norm=norm, random_start=random_start, device=device)
    elif attack == "fgsm" :
        adv = FGSM(model, bound=bound, norm=norm, random_start=random_start, device=device)
    elif attack == "ball" :
        adv = EpsilonAdversary(model, epsilon=bound, repeat=repeat_num, norm=norm, device=device)

    return adv, stats


def trainer(model, logger, loader, args, data="mnist", optimizer=None, scheduler=None, adv_train=None, tboard=True, **kwargs) :
    # loader : train_loader, train_eval_loader, test_loader
    logger.info("="*80)
    logger.info("Train Info")
    logger.info("Model : {}".format(args.model))
    logger.info("Number of blocks : {}".format(args.block))
    logger.info("Number of parameters : {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    start_time = time.time()
    best_acc = 0.
    device = args.device

    criterion = nn.CrossEntropyLoss().to(args.device)
    logger.info("Criterion : {}".format(criterion.__class__.__name__))
    logger.info("Adversarial Training : {}".format(adv_train))
    logger.info("="*80)
    data_gen = inf_generator(loader['train_loader'])
    batches_per_epoch = len(loader['train_loader'])

    best_acc = 0.
    best_loss = 1000.
    best_acc_epoch = 0
    best_loss_epoch = 0
    batch_time_meter = RunningAverageMeter()
    end_time = time.time()
    if args.hist :
        hist_dict = dict()

    adv, stats = adv_train_module(adv_train, model, data, args.iters, args.device)

    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_acc.pt"))
    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_loss.pt"))
    for itr in tqdm.tqdm(range(args.epochs * batches_per_epoch)) :
        
        if itr % batches_per_epoch == 0 and scheduler is not None :
            scheduler.step()

        model.train()
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(args.device)
        y = y.to(args.device)

        if adv_train is not None :
            x = adv.perturb(x,y,device=args.device)
            if adv_train == "ball" :
                y = torch.cat([y for _ in range(stats[0])])
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)

        loss.backward()
        optimizer.step()
        
        batch_time_meter.update(time.time() - end_time)
        end_time = time.time()

        if itr % batches_per_epoch == 0 :
            model.eval()
            with torch.no_grad() :
                train_acc, train_loss = accuracy(model, dataset_loader=loader['train_eval_loader'], device=args.device, criterion=criterion)
                val_acc, val_loss = accuracy(model, dataset_loader=loader['test_loader'], device=args.device, criterion=criterion)
                if val_acc >= best_acc :
                    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_acc.pt"))
                    best_acc = val_acc
                    best_acc_epoch = int(itr // batches_per_epoch)
                if val_loss <= best_loss :
                    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_loss.pt"))
                    best_loss = val_loss
                    best_loss_epoch = int(itr // batches_per_epoch)
                logger.info(
                        "Epoch {:03d} | Time {:.3f} ({:.3f}) | Train loss {:.4f} | Validation loss {:.4f} | Train Acc {:.4f} | Validation Acc {:.4f}".format(
                            int(itr // batches_per_epoch), batch_time_meter.val, batch_time_meter.avg, train_loss, val_loss, train_acc, val_acc
                            )
                        )
            torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_final.pt"))

    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_final.pt"))
    if args.hist :
        with open(os.path.join(args.save,"history.json"),"w") as f :
            json.dump(hist_dict,f)

    logger.info("="*80)
    logger.info("Required Time : {:03d} minute {:.2f} seconds".format(int((time.time()-start_time) // 60), (time.time()-start_time) % 60))
    logger.info("Best Acc Epoch : {:03d}".format(best_acc_epoch))
    logger.info("Best Validation Accuracy : {:.4f}".format(best_acc))
    logger.info("Best loss Epoch : {:03d}".format(best_loss_epoch))
    logger.info("Best Validation loss : {:.4f}".format(best_loss))
    logger.info("Train end")
    logger.info("="*80)

    return model

def test(model, target_loader, device, **kwargs) :
    model.eval()
    with torch.no_grad() :
        eval_acc, eval_loss = accuracy(model, dataset_loader=target_loader, device=device, criterion=nn.CrossEntropyLoss().to(device))
    return eval_acc, eval_loss

# The below function is not used in our experiment.
def adversarial_attack(model, logger, target_loader, save_adv, args, **kwargs) :
    logger.info("="*80)
    logger.info("Natural result")
    orig_acc, orig_loss = test(model, target_loader, args.device)
    logger.info("Accuracy : {:.4f}".format(orig_acc))
    logger.info("Loss : {:.4f}".format(orig_loss))
    logger.info("-"*80)
    logger.info("Attack parameters : eps={}".format(args.eps))

    repeat = 1
    if args.attack == "FGSM" :
        attack_module = FGSM(model, epsilon=args.eps, device=args.device)
    elif args.attack == "bim" :
        k = 10
        attack_module = LinfPGD(model, epsilon=args.eps, alpha=args.eps/k, k=k, random_start=False, device=args.device)
    elif args.attack == "mim" :
        k = 10
        attack_module = MIM(model, epsilon=args.eps, alpha=args.eps/k, k=k, device=args.device)
    elif args.attack == "pgd" :
        k = 10
        attack_module = LinfPGD(model, epsilon=args.eps, alpha=args.eps/k, k=k, device=args.device)
    elif args.attack == "ball" :
        repeat = 5
        attack_module = EpsilonAdversary(model, epsilon=args.eps, repeat=repeat, dist="uniform", device=args.device)
   
    criterion = nn.CrossEntropyLoss().to(args.device)
    total_correct = 0
    if save_adv :
        save_adv = os.path.join(args.load, args.attack+"_"+str(args.eps))
    else :
        save_adv = None
    adv_acc,_ = accuracy(model, dataset_loader=target_loader, save_adv=save_adv, repeat=repeat, device=args.device, criterion=criterion, attack=attack_module)
    logger.info("Attacked Accuracy : {:.4f}".format(adv_acc))
    logger.info("Finished")
    logger.info("="*80)
