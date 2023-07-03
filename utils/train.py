import torch
from utils.DDP import is_main_process,reduce_value
from tqdm import tqdm
import sys
from mod.loss import MulTaskLoss
def train_one_epoch(model,
                    MaskLossMode,
                    optimizer,
                    data_loader,
                    device,
                    epoch):
    model.train()
    #mean_loss = torch.zeros(1).to(device)
    LOSS1 = 0.0
    LOSS2 = 0.0
    LOSS3 = 0.0
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        train_inputs = data[0].float().to(device)
        pf_gt = data[1].float().to(device)
        mask_gt = data[2].long().to(device)

        pred = model(train_inputs)

        loss,metrics= MulTaskLoss(pred, (pf_gt,mask_gt),model_input=train_inputs,gamma=0.5,pfmode=1,normalization=32,MaskLossMode=MaskLossMode)
        loss.backward()
        loss = reduce_value(loss, average=True)
        #mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        LOSS1 = (LOSS1 * step + metrics["pfloss"]) / (step + 1)
        LOSS2 = (LOSS2 * step + metrics["maskloss"]) / (step + 1)
        # LOSS3 = (LOSS3 * step + metrics["SSIM"]) / (step + 1)
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "Epoch[{}]".format(epoch)
            data_loader.set_postfix(metrics)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    if is_main_process():
        print(" Epoch[{:0>3}]  SL1loss: {:.4f} CEloss:{:.4f} SSIM: {:.4f}".format(
                epoch + 1,   LOSS1 , LOSS2 , LOSS3  ))
    return LOSS1,LOSS2