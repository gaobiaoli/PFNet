import torch
import torch.nn as nn
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w=x.size()
        avgout = self.shared_MLP(self.avg_pool(x).view([b,c]))
        maxout = self.shared_MLP(self.max_pool(x).view([b,c]))
        output=self.sigmoid(avgout + maxout)
        output=output.view([b,c,1,1])
        return output


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding=kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel,CAreduction=16,SAkernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel,reduction=CAreduction)
        self.spatial_attention = SpatialAttention(kernel_size=SAkernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

if __name__=="__main__":
    input=torch.randn([16,128,64,64]).cuda()
    model=CBAM(128).cuda()
    output=model(input)
