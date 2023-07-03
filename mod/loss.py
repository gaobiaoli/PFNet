import torch
import torch.nn as nn
import torch.nn.functional as F
from MSCNet import MSCNet
from utils.eval import EvalCocoDataset,evaluate_PFNet
from mod.transform import warp
from mod.ssim import SSIM
def CE_Loss(inputs, target):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss()(temp_inputs, temp_target)
    return CE_loss

def MaskLoss(mask_pre,mask_gt,LossMode="BCE"):
    if LossMode=="BCE":
        assert mask_pre.shape[1]<=mask_pre.shape[1]
        loss=0
        for i in range(mask_pre.shape[1]):
            loss += nn.BCEWithLogitsLoss()(mask_pre[:,i,:,:],mask_gt[:,i,:,:].float())
        return loss
    elif LossMode=='CrossEntropy':
        assert mask_pre.shape[1]%2==0
        loss = 0
        for i in range(mask_pre.shape[1]//2):
            loss += nn.CrossEntropyLoss()(mask_pre[:, 2*i:2*i+2, :, :], mask_gt[:, i, :, :].long())
        return loss


def MulTaskLoss(outputs,gt,model_input=None,gamma=0.5,pfmode=1,normalization=1,MaskLossMode='BCE'):
    """
    pfmode: 0 不使用mask计算PFloss
            1 使用mask_gt计算PFloss
    """
    pf_outputs, mask_outputs=outputs
    pf_gt, mask_gt=gt
    pixel_count=torch.sum(mask_gt[:, 0, :, :])
    pf_mask_gt=mask_gt[:, 0, :, :]
    if pfmode == 0:
        pfloss=nn.SmoothL1Loss(reduction="mean")(pf_outputs*normalization, pf_gt)
    elif pfmode == 1:
        pfloss=nn.SmoothL1Loss(reduction="sum")(pf_outputs*pf_mask_gt.unsqueeze(1)*normalization
                                 , pf_gt*pf_mask_gt.unsqueeze(1))/pixel_count
    # ssimloss=SSIMLoss(model_input,pf_outputs,mask_gt)
    maskloss=MaskLoss(mask_outputs,mask_gt,LossMode=MaskLossMode)
    metrics={"pfloss":pfloss.item(),"maskloss":maskloss.item()
        # ,"SSIM":-ssimloss.item()
             }
    return pfloss*gamma+(1-gamma)*maskloss,metrics

def SSIMLoss(input,pf_outputs,mask_gt):
    B, _, H, W = input.size()
    raw_img=input[:,0,:,:]
    gt_warped_img=input[:,1,:,:]
    raw_mask=mask_gt[:,0,:,:]
    warped_mask=mask_gt[:,1,:,:]
    pre_warped_img=warp((raw_img*raw_mask).unsqueeze(1),pf_outputs)
    ssimLoss=-SSIM(mask_average=True)(gt_warped_img.unsqueeze(1),pre_warped_img,warped_mask.unsqueeze(1))
    return ssimLoss



