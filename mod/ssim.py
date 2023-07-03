import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, mask_average=True,mask=None):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask_average and mask is not None:
            return (ssim_map*mask).sum()/mask.sum()
        else:
            return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, mask_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.mask_average = mask_average
        self.channel = 1
        self.window = create_window(window_size, self.channel).cuda()

    def forward(self, img1, img2,mask=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average,self.mask_average,mask=mask)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from mod.transform import warp
    from MSCNet import MSCNet
    from PIL import Image
    from utils.eval import EvalCocoDataset

    weight_path = r"F:\history\logs\MSCNet\2-9-base\pfnet_0080.pth"
    model = MSCNet()
    model.eval()
    model.load_state_dict(torch.load(weight_path))
    dataset_val = EvalCocoDataset(r"F:\code_python\PFNet-pytorch", "results", year="")
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    a = val_loader.dataset[0]

    pre = model(torch.tensor(a[0]).float().unsqueeze(0))
    mask_pred = F.softmax(pre[1][0], dim=0).argmax(dim=0).numpy()
    mask_gt = a[7]
    ori = (torch.tensor(a[0][0]).unsqueeze(0).unsqueeze(0)
           * mask_gt[0]
           ).float()
    pre_warped_image = warp(ori.float(), pre[0] * 32)[0][0]
    gt_warped_image = torch.tensor(a[7][1, :, :] * a[0][1])
    #Image.fromarray(gt_warped_image.numpy() * 255).show()
    #Image.fromarray(pre_warped_image.detach().numpy() * 255).show()
    ssim = SSIM()
    i1=pre_warped_image.unsqueeze(0).unsqueeze(0)
    i2=gt_warped_image.unsqueeze(0).unsqueeze(0).float()
    a=ssim(i1,i2,mask_gt[1])
    print(a)
