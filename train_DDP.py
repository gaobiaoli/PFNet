import time
import numpy as np
import argparse
import random
import os
import tempfile
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
from utils.train import train_one_epoch
from mod.loss import CE_Loss,MulTaskLoss
from utils.DDP import cleanup,dist,init_distributed_mode
from evaluate import evaluate_PFNet,EvalCocoDataset
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    #args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = "initial.pth"

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(logdir=args.logs)

    # 实例化训练数据集
    train_dataset = CocoDataset(args.dataset, "train", year=args.year,
                                diffusion=None)

    # 实例化验证数据集


    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset, num_workers=nw, collate_fn=safe_collate
                              ,pin_memory=True,batch_sampler=train_batch_sampler)


    # 实例化模型
    model = MSCNet().to(device)
    # 如果存在预训练权重则载入

    #checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重

        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    # Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # params1 = list(map(id, model.module.FeatureExtraction.parameters()))
    # base_params = filter(lambda p: id(p) not in params1, model.module.parameters())
    # optimizer = optim.Adam([{"params": base_params, "lr": args.lr},
    #     {"params": model.module.FeatureExtraction.parameters(), "lr": 1e-3}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    for epoch in range(args.n_epoch):
        train_sampler.set_epoch(epoch)

        pf_loss,mask_loss = train_one_epoch(model=model,
                                    MaskLossMode='CrossEntropy',
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        if rank == 0:
            tb_writer.add_scalar("pfloss", pf_loss, epoch)
            tb_writer.add_scalar("maskloss", mask_loss, epoch)
            # tb_writer.add_scalar("SSIM", SSIM, epoch)
            if (epoch + 1) % 10 == 0:
                filename = 'pfnet_{:0>4}.pth'.format(epoch + 1)
                torch.save(model.module.state_dict(), os.path.join(args.logs, filename))
    # 删除临时缓存文件

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4*5, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='set the number of epochs, default is 200')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='set the batch size, default is 32.')
    parser.add_argument('--seed', type=int, default=1987,
                        help='Pseudo-RNG seed')

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    #parser.add_argument("--local_rank", default=-1, type=int)
    opt = parser.parse_args()

    main(opt)