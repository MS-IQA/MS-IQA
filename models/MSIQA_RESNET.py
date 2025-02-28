import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Net(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Net, self).__init__()

        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = self.resnet50.avgpool
        self.fc = self.resnet50.fc

    def forward(self, x):
        # save the outpus of intermediate stages
        outputs = {}

        y = self.resnet50(x)

        # Conv1, BN1, ReLU, MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage1
        x = self.layer1(x)
        outputs['Stage1'] = x

        # Stage2
        x = self.layer2(x)
        outputs['Stage2'] = x

        # Stage3
        x = self.layer3(x)
        outputs['Stage3'] = x

        # Stage4
        x = self.layer4(x)
        outputs['Stage4'] = x

        # Global Average Pooling
        x = self.avgpool(x)

        # Flatten and Fully Connected Layer (FC)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return outputs