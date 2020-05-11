from torchvision.models.resnet import ResNet, BasicBlock,resnet152
import torch

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        #[2, 2, 2, 2]for resnet18, [3, 4, 6, 3] for resnet34
        self.conv1 = torch.nn.Conv2d(1, 64, 
                                    kernel_size=(7, 7), 
                                    stride=(2, 2), 
                                    padding=(3, 3), bias=False)