
import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.InstanceNorm2d(out_channels, track_running_stats=False, affine=True)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels, track_running_stats=False, affine=True)
        self.bn2 = nn.InstanceNorm2d(out_channels, track_running_stats=False, affine=True)

    def forward(self, X):
        shortcut = self.shortcut(X)
        X = nn.ReLU()(self.bn1(self.conv1(X)))
        X = nn.ReLU()(self.bn2(self.conv2(X)))
        X = X + shortcut
        return nn.ReLU()(X)
    


class ResNet34(nn.Module):
    def __init__(self, in_channels, resblock, outputs=6):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, track_running_stats=False, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)
        X2 = X[:,2].reshape(X.shape[0], 1, 512, 512)
        X3 = X[:,3].reshape(X.shape[0], 1, 512, 512)
        X4 = X[:,4].reshape(X.shape[0], 1, 512, 512)
        X5 = X[:,5].reshape(X.shape[0], 1, 512, 512)

        X0 = self.layer0(X0)
        X0 = self.layer1(X0)
        X0 = self.layer2(X0)
        X0 = self.layer3(X0)
        X0 = self.layer4(X0)
        X0 = self.gap(X0)
        X0 = X0.reshape(X0.shape[0],512)

        X1 = self.layer0(X1)
        X1 = self.layer1(X1)
        X1 = self.layer2(X1)
        X1 = self.layer3(X1)
        X1 = self.layer4(X1)
        X1 = self.gap(X1)
        X1 = X1.reshape(X1.shape[0],512)

        X2 = self.layer0(X2)
        X2 = self.layer1(X2)
        X2 = self.layer2(X2)
        X2 = self.layer3(X2)
        X2 = self.layer4(X2)
        X2 = self.gap(X2)
        X2 = X2.reshape(X2.shape[0],512)

        X3 = self.layer0(X3)
        X3 = self.layer1(X3)
        X3 = self.layer2(X3)
        X3 = self.layer3(X3)
        X3 = self.layer4(X3)
        X3 = self.gap(X3)
        X3 = X3.reshape(X3.shape[0],512)

        X4 = self.layer0(X4)
        X4 = self.layer1(X4)
        X4 = self.layer2(X4)
        X4 = self.layer3(X4)
        X4 = self.layer4(X4)
        X4 = self.gap(X4)
        X4 = X4.reshape(X4.shape[0],512)

        X5 = self.layer0(X5)
        X5 = self.layer1(X5)
        X5 = self.layer2(X5)
        X5 = self.layer3(X5)
        X5 = self.layer4(X5)
        X5 = self.gap(X5)
        X5 = X5.reshape(X5.shape[0],512)

        X = torch.cat((X0, X1, X2, X3, X4, X5), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet34Pl2(nn.Module):
    def __init__(self, in_channels, resblock, outputs=6):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, track_running_stats=False, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)

        X0 = self.layer0(X0)
        X0 = self.layer1(X0)
        X0 = self.layer2(X0)
        X0 = self.layer3(X0)
        X0 = self.layer4(X0)
        X0 = self.gap(X0)
        X0 = X0.reshape(X0.shape[0],512)

        X1 = self.layer0(X1)
        X1 = self.layer1(X1)
        X1 = self.layer2(X1)
        X1 = self.layer3(X1)
        X1 = self.layer4(X1)
        X1 = self.gap(X1)
        X1 = X1.reshape(X1.shape[0],512)

        X = torch.cat((X0, X1), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=6):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, track_running_stats=False, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)
        X2 = X[:,2].reshape(X.shape[0], 1, 512, 512)
        X3 = X[:,3].reshape(X.shape[0], 1, 512, 512)
        X4 = X[:,4].reshape(X.shape[0], 1, 512, 512)
        X5 = X[:,5].reshape(X.shape[0], 1, 512, 512)

        X0 = self.layer0(X0)
        X0 = self.layer1(X0)
        X0 = self.layer2(X0)
        X0 = self.layer3(X0)
        X0 = self.layer4(X0)
        X0 = self.gap(X0)
        X0 = X0.reshape(X0.shape[0],512)

        X1 = self.layer0(X1)
        X1 = self.layer1(X1)
        X1 = self.layer2(X1)
        X1 = self.layer3(X1)
        X1 = self.layer4(X1)
        X1 = self.gap(X1)
        X1 = X1.reshape(X1.shape[0],512)

        X2 = self.layer0(X2)
        X2 = self.layer1(X2)
        X2 = self.layer2(X2)
        X2 = self.layer3(X2)
        X2 = self.layer4(X2)
        X2 = self.gap(X2)
        X2 = X2.reshape(X2.shape[0],512)

        X3 = self.layer0(X3)
        X3 = self.layer1(X3)
        X3 = self.layer2(X3)
        X3 = self.layer3(X3)
        X3 = self.layer4(X3)
        X3 = self.gap(X3)
        X3 = X3.reshape(X3.shape[0],512)

        X4 = self.layer0(X4)
        X4 = self.layer1(X4)
        X4 = self.layer2(X4)
        X4 = self.layer3(X4)
        X4 = self.layer4(X4)
        X4 = self.gap(X4)
        X4 = X4.reshape(X4.shape[0],512)

        X5 = self.layer0(X5)
        X5 = self.layer1(X5)
        X5 = self.layer2(X5)
        X5 = self.layer3(X5)
        X5 = self.layer4(X5)
        X5 = self.gap(X5)
        X5 = X5.reshape(X5.shape[0],512)

        X = torch.cat((X0, X1, X2, X3, X4, X5), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet18Pl2(nn.Module):
    def __init__(self, in_channels, resblock, outputs=6):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, track_running_stats=False, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)

        X0 = self.layer0(X0)
        X0 = self.layer1(X0)
        X0 = self.layer2(X0)
        X0 = self.layer3(X0)
        X0 = self.layer4(X0)
        X0 = self.gap(X0)
        X0 = X0.reshape(X0.shape[0],512)

        X1 = self.layer0(X1)
        X1 = self.layer1(X1)
        X1 = self.layer2(X1)
        X1 = self.layer3(X1)
        X1 = self.layer4(X1)
        X1 = self.gap(X1)
        X1 = X1.reshape(X1.shape[0],512)

        X = torch.cat((X0, X1), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output

