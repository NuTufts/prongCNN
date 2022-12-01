
import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, bnMom=0.1):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bnMom)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bnMom)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bnMom)

    def forward(self, X):
        shortcut = self.shortcut(X)
        X = nn.ReLU()(self.bn1(self.conv1(X)))
        X = nn.ReLU()(self.bn2(self.conv2(X)))
        X = X + shortcut
        return nn.ReLU()(X)
    


class ResNet34(nn.Module):
    def __init__(self, in_channels, resblock, outputs=5, batchMom=0.1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=batchMom),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)
        X2 = X[:,2].reshape(X.shape[0], 1, 512, 512)

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

        X = torch.cat((X0, X1, X2), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet34Pl2(nn.Module):
    def __init__(self, in_channels, resblock, outputs=5, batchMom=0.1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=batchMom),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X = self.layer0(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.gap(X)
        X = X.reshape(X.shape[0],512)

        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=5, batchMom=0.1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=batchMom),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, X):

        X0 = X[:,0].reshape(X.shape[0], 1, 512, 512)
        X1 = X[:,1].reshape(X.shape[0], 1, 512, 512)
        X2 = X[:,2].reshape(X.shape[0], 1, 512, 512)

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

        X = torch.cat((X0, X1, X2), 1)
        X = self.fc(X)
        output = self.logSoftmax(X)

        return output



class ResNet18Pl2(nn.Module):
    def __init__(self, in_channels, resblock, outputs=5, batchMom=0.1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=batchMom),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False, bnMom=batchMom),
            resblock(64, 64, downsample=False, bnMom=batchMom)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, bnMom=batchMom),
            resblock(128, 128, downsample=False, bnMom=batchMom)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, bnMom=batchMom),
            resblock(256, 256, downsample=False, bnMom=batchMom)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, bnMom=batchMom),
            resblock(512, 512, downsample=False, bnMom=batchMom)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, outputs)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        #print("input shape:", X.shape)
        X = self.layer0(X)
        #print("shape after layer0:", X.shape)
        X = self.layer1(X)
        #print("shape after layer1:", X.shape)
        X = self.layer2(X)
        #print("shape after layer2:", X.shape)
        X = self.layer3(X)
        #print("shape after layer3:", X.shape)
        X = self.layer4(X)
        #print("shape after layer4:", X.shape)
        X = self.gap(X)
        #print("shape after gap:", X.shape)
        X = X.reshape(X.shape[0],512)
        #print("shape after reshape:", X.shape)
        X = self.fc(X)
        #print("shape after fc:", X.shape)
        output = self.logSoftmax(X)

        return output

