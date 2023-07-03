import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import math
import matplotlib.pyplot as plt
def getheta(theta):
    def sin(theta):
        return math.sin(theta)
    def cos(theta):
        return math.cos(theta)
    return torch.tensor([[cos(theta),sin(theta),1],[-sin(theta),cos(theta),1]])


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if float(torch.__version__[:3]) >= 1.3:
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    else:
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output
# def perspectiveTransform(H,size):
#     F.affine_grid(60,(128,128))
def flowreset(H,flo):
    pass
if __name__=="__main__":
    theta=getheta(3.14/4).unsqueeze(0)
    img = cv2.imread(r'F:\code_python\PFNet-pytorch\results\1.jpg', cv2.IMREAD_GRAYSCALE)
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(theta, size=img.shape)
    output = F.grid_sample(img, grid)[0].numpy().transpose(1, 2, 0).squeeze()
    plt.subplot(2, 1, 2)
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.show()