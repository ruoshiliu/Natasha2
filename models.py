import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock,resnet152

class MnistLeNet(nn.Module):
    def __init__(self):
        super(MnistLeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.conv2 = nn.Conv2d(2, 3, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 6 * 6, 5)  # 6*6 from image dimension
        self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        #[2, 2, 2, 2]for resnet18, [3, 4, 6, 3] for resnet34
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
class CifarLeNet(nn.Module):
    def __init__(self):
        super(CifarLeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 2, 3)
        self.conv2 = nn.Conv2d(2, 3, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 6 * 6, 5)  # 6*6 from image dimension
        self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CifarResNet(ResNet):
    def __init__(self):
        super(CifarResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        #[2, 2, 2, 2]for resnet18, [3, 4, 6, 3] for resnet34
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
