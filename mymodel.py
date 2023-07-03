import torch
import torch.nn as nn
from torchvision import models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class GroupDilatedBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(GroupDilatedBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=2, padding=2),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=4, padding=4),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=6, padding=6),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=8, padding=8),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv1x1 = nn.Sequential(conv1x1(planes*4, planes),
                                     nn.BatchNorm2d(planes),
                                     #nn.ReLU(inplace=True)
                                     )
    def forward(self,x):
        out=[self.conv1(x),self.conv2(x),self.conv3(x),self.conv4(x)]
        return self.conv1x1(torch.cat(out,dim=1))
class GroupDeconvBlock(nn.Module):
    def __init__(self, inplanes,planes):
        super(GroupDeconvBlock, self).__init__()
        self.convt1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.convt2 = nn.Sequential(
                                    nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=2,padding=1,output_padding=1),
                                    nn.BatchNorm2d(planes),
                                    #nn.ReLU(inplace=True)
                                    )
        self.GroupBlock  = GroupDilatedBlock(planes, planes)
        self.Last=nn.ReLU(inplace=True)


    def forward(self, x):
        out1=self.convt1(x)
        out2=self.convt2(x)
        out2=self.GroupBlock(out2)



        return self.Last(out1+out2)
class ConvBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes * self.expansion, stride),
            nn.BatchNorm2d(planes * self.expansion),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)

        return self.relu(out)
class IdentityBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes):
        super(IdentityBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x

        return self.relu(out)
class MyPFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MyPFNet, self).__init__()
        self.pretrained = pretrained

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ConvBlock(64, 64),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(256, 128, stride=2),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128)
        )
        # self.layer3 = nn.Sequential(
        #     ConvBlock(512, 256, stride=2),
        #     IdentityBlock(1024, 256),
        #     IdentityBlock(1024, 256),
        #     IdentityBlock(1024, 256),
        #     IdentityBlock(1024, 256),
        #     IdentityBlock(1024, 256)
        # )

        # self.de_layer1 = nn.Sequential(
        #     GroupDeconvBlock(1024, 512),
        #     IdentityBlock(512, 128),
        #     IdentityBlock(512, 128),
        #     IdentityBlock(512, 128)
        # )
        self.de_layer2 = nn.Sequential(
            GroupDeconvBlock(512, 256),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        self.de_layer3 = nn.Sequential(
            GroupDeconvBlock(256, 128),
            IdentityBlock(128, 32)
        )
        self.de_layer4 = GroupDeconvBlock(128, 64)

        self.out = nn.Sequential(
            conv1x1(64, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        new_state_dict = {}
        pretrained_dict = models.resnet50(pretrained=True)
        model_dict = self.state_dict()
        for k, v in pretrained_dict.state_dict().items():
            if k in model_dict.keys() and k != 'conv1.weight':
                new_state_dict[k] = v
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)

        # x = self.de_layer1(x)
        x = self.de_layer2(x)
        x = self.de_layer3(x)
        x = self.de_layer4(x)
        out = self.out(x)

        return out
if __name__=="__main__":
    from torchsummary import summary

    net = MyPFNet()
    #net = DeconvBlock(8, 4)
    # print(net)
    net.cuda()
    summary(net, (2, 128, 128))