import time
import numpy as np
import argparse
import random
import os
#os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import CocoDataset, safe_collate
from model import PFNet
from torchsummary import summary
from MSCNet import MSCNet
from mod.loss import CE_Loss,MulTaskLoss
from evaluate import evaluate_PFNet,EvalCocoDataset
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Perspective Network on COCO.')
    #Paths
    parser.add_argument('--dataset', required=False,
                        default="/CV/3T/dataset-public/COCO/coco2014",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default="2014",
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--logs', required=False,
                        default="./logs",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4*4, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=130,
                        help='set the number of epochs, default is 200')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='set the batch size, default is 32.')
    parser.add_argument('--seed', type=int, default=1987,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    validation = False
    evaluate = True
    normalize_factor=32
    # Data ####################################################################
    train_dataset = CocoDataset(args.dataset, "train", year=args.year,
                                diffusion=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=safe_collate,pin_memory=True)
    if validation:
        val_dataset = CocoDataset(args.dataset, "val", year=args.year)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=safe_collate)
    if evaluate:
        eval_dataset=EvalCocoDataset(args.dataset, "val", year=args.year)
        eval_loader=DataLoader(eval_dataset, batch_size=50, shuffle=False)
    #print(len(train_dataset), len(train_loader), len(val_dataset), len(val_loader))

    # Initialize network ######################################################
    model = MSCNet()

    GPUs=True


    if GPUs:
        model=nn.DataParallel(model,device_ids=[0]).cuda()
    else:
        model = model.to(device)
    #summary(model, (2, 128, 128))
    params1 = list(map(id, model.module.FeatureExtraction.parameters()))
    base_params = filter(lambda p: id(p) not in params1, model.module.parameters())
    optimizer = optim.Adam([{"params": base_params,"lr": args.lr},
                            {"params": model.module.FeatureExtraction.parameters(), "lr": 1e-3}])
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    pf_loss = nn.SmoothL1Loss()
    mask_loss=nn.CrossEntropyLoss()

    # Training ################################################################
    # check path
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    writer = SummaryWriter(logdir=args.logs)

    print("start training")
    for epoch in range(args.n_epoch):
        # train
        model.train()
        LOSS1 = 0.0
        LOSS2 = 0.0
        with tqdm(total=len(train_loader),desc="Train") as pdar:
            for i, batch_value in enumerate(train_loader):
                glob_iter = epoch * len(train_loader) + i

                train_inputs = batch_value[0].float().to(device)
                pf_gt = batch_value[1].float().to(device)
                mask_gt = batch_value[2].long().to(device)
                optimizer.zero_grad()
                pred = model(train_inputs)
                loss, metrics = MulTaskLoss(pred, (pf_gt, mask_gt),model_input=train_inputs,pfmode=1,MaskLossMode='CrossEntropy')
                # loss=MulTaskLoss(train_outputs * normalize_factor, mask_outputs,
                #                  train_gt,mask_gt)
                # loss1 = pf_loss(train_outputs * normalize_factor * mask_gt.unsqueeze(1), pf_gt * mask_gt.unsqueeze(1))
                # loss2 = mask_loss(mask_outputs, mask_gt)
                #loss=0.5*loss1+0.5*loss2
                loss.backward()
                optimizer.step()

                #mask_loss += loss2.item()
                LOSS1 += metrics["pfloss"]
                LOSS2 += metrics["maskloss"]
                pdar.set_postfix(metrics)
                pdar.update(1)
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>4}]/[{:0>4}] SL1loss: {:.4f} CEloss:{:.4f} lr={:.8f}".format(
                        epoch + 1, args.n_epoch, i + 1, len(train_loader), LOSS1 / len(train_loader), LOSS2/ len(train_loader),scheduler.get_last_lr()[0]))
            scheduler.step()
        LOSS1 /= len(train_loader)
        LOSS2 /= len(train_loader)
        writer.add_scalar('SL1_loss',LOSS1, epoch)
        writer.add_scalar('CE_loss', LOSS2, epoch)
        time.sleep(1)

        # validation
        if validation:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, batch_value in enumerate(val_loader):
                    val_inputs = batch_value[0].float().to(device)
                    val_gt = batch_value[1].float().to(device)

                    val_outputs = model(val_inputs)
                    #loss = smooth_l1_loss(val_outputs, val_gt)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        # evaluate
        if evaluate:
            model.eval()
            with torch.no_grad():
                eval_loader.dataset.rho=train_loader.dataset.rho
                evaluate_PFNet(model, eval_loader, 10, device,
                               normalize_factor=32,macePlot=False)
            time.sleep(1)

        # save model
        if (epoch+1) % 10 == 0:
            filename = 'pfnet_{:0>4}.pth'.format(epoch + 1)
            torch.save(model, os.path.join(args.logs, filename))
        # update diffusion
        if (epoch + 1) % 5 == 0:
            train_loader.dataset.step()
    print("Finished training.")
