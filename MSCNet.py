import torch
import torch.nn as nn
from model import conv1x1,conv3x3
from torchsummary import summary
from mod.attention.CBAM import CBAM
from mod.encoder.ASPP import ASPP
from mod.attention.AxialAttention import AxialBlock
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class GroupDilatedBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(GroupDilatedBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=2, padding=2,bias=False),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=4, padding=4,bias=False),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=6, padding=6,bias=False),
                                   nn.BatchNorm2d(planes, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, dilation=8, padding=8,bias=False),
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
                                    nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=2,padding=1,output_padding=1,bias=False),
                                    nn.BatchNorm2d(planes),
                                    )
        self.GroupBlock  = GroupDilatedBlock(planes, planes)
        self.Last=nn.ReLU(inplace=True)


    def forward(self, x):
        out1=self.convt1(x)
        out2=self.convt2(x)
        out2=self.GroupBlock(out2)



        return self.Last(out1+out2)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,dowmsample=None):
        super(ResidualBlock, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        if stride!=1 or in_channels!=out_channels:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample=dowmsample
    def forward(self,x):
        identity=x
        out=self.layer1(x)
        out=self.layer2(out)
        if self.downsample is not None:
            identity=self.downsample(identity)
        out+=identity
        return self.relu(out)

class FeatureExtraction(nn.Module):
    def __init__(self,normalization=True,GlobalFeature=True):
        super(FeatureExtraction, self).__init__()
        self.GlobalFeature=GlobalFeature
        self.ConvBlock=nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=64,kernel_size=7,padding=3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            # ResidualBlock(64, 64, 1),
            AxialBlock(64, 32, kernel_size=64),
            # AxialBlock(64, 32, kernel_size=64),
            # AxialBlock(64, 32, kernel_size=64),
            # AxialBlock(64, 32, kernel_size=64),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1),
            # AxialBlock(64, 64, kernel_size=64,stride=2),
            # AxialBlock(128, 64, kernel_size=32),
            # AxialBlock(128, 64, kernel_size=32),
            AxialBlock(128, 64, kernel_size=32)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1),
            # ResidualBlock(256, 256, 1),
            # AxialBlock(128, 128, kernel_size=32,stride=2),
            # AxialBlock(256, 128, kernel_size=16),
            # AxialBlock(256, 128, kernel_size=16),
            AxialBlock(256, 128, kernel_size=16)
        )
        # self.layer4 = nn.Sequential(
        #     ResidualBlock(256, 512, 2),
        #     ResidualBlock(512, 512, 1),
        # )
    def forward(self,x):
        f1=self.layer1(self.ConvBlock(x))
        f2=self.layer2(f1)
        f3=self.layer3(f2)
        #f4=self.layer4(f3)

        return [f1,f2,f3]


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True,outputflatten=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.outputflatten = outputflatten
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, MSFeatureList):
        corrList=[]
        for i in range(len(MSFeatureList)):
            feature_A=MSFeatureList[i].chunk(2,1)[0]
            feature_B=MSFeatureList[i].chunk(2,1)[1]
            if self.shape == '3D':
                b, c, h, w = feature_A.size()
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
                feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B, feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
            elif self.shape == '4D':
                b, c, hA, wA = feature_A.size()
                b, c, hB, wB = feature_B.size()
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
                feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A, feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b, hA, wA, hB*wB)

            if self.outputflatten:
                correlation_tensor = correlation_tensor.view(b, hA, wA, hB * wB).transpose(2, 3).transpose(1, 2)

            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

            corrList.append(correlation_tensor)
        return corrList

class PFMaskNet(nn.Module):
    def __init__(self,input_channel):
        super(PFMaskNet, self).__init__()
        self.attention=CBAM(input_channel)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(input_channel, 48, 1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.upsample=nn.Sequential(
                    GroupDeconvBlock(48,24),
                    GroupDeconvBlock(24,12))
        self.out=nn.Sequential(
                nn.Conv2d(in_channels=12,out_channels=4,
                          kernel_size=3,padding=1,bias=True))
    def forward(self,x,output_size):
        H,W=output_size
        x = self.attention(x)
        x = self.shortcut_conv(x)
        x = self.upsample(x)
        x = self.out(x)
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class MaskNet(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(MaskNet, self).__init__()
        self.attention=CBAM(input_channels,CAreduction=8)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.out=nn.Sequential(
                nn.Conv2d(in_channels=32,out_channels=output_channels,
                          kernel_size=1))
    def forward(self,x):
        x = self.attention(x)
        x = self.shortcut_conv(x)
        x = self.out(x)
        return x

class MSCNet(nn.Module):
    def __init__(self):
        super(MSCNet, self).__init__()
        self.FeatureExtraction=FeatureExtraction(GlobalFeature=False)
        self.FeatureCorrelation=FeatureCorrelation(shape="4D",normalization=True)
        self.ASPP=ASPP(256,[3,6,9])
        self.upsample1=nn.Sequential(
            #GroupDeconvBlock(512,256),
            GroupDeconvBlock(256,128)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(256, 128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)
        self.attention1=CBAM(128)
        self.upsample2=nn.Sequential(
            GroupDeconvBlock(128,64),
            GroupDeconvBlock(64,32)
        )
        self.attention2=CBAM(32,CAreduction=8)
        self.out = nn.Sequential(
            conv1x1(32, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1)
        )
        self.maskNet=PFMaskNet(256)
    def forward(self,x):
        b,c,h,w=x.shape

        MSCFeature=self.FeatureExtraction(x)
        mask=self.maskNet(MSCFeature[2],(h,w))
        GlobalFeature=MSCFeature[-1]
        out=GlobalFeature
        out=self.ASPP(out)
        out=self.upsample1(out)
        if True:
            out=torch.cat((out,MSCFeature[-2]),dim=1)
            out=self.conv1x1(out)
        out=self.attention1(out)
        out=self.upsample2(out)
        out=self.attention2(out)
        out=self.out(out)
        return (out,mask)
        #MSCCorr=self.FeatureCorrelation(MSCFeature)


if __name__=="__main__":
    model=MSCNet().cuda()
    #model = torchvision.models.resnet50()
    # a=torch.randn([16,2,128,128]).cuda()
    # b=model(a)
    summary(model,(6,128,128),batch_size=32)





