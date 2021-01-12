import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block for resnet 18"""

    expansion = 1


    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual block
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1),
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride))

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet(nn.Module):

    def __init__(self, block, num_block, in_channels, up_scale):
        super().__init__()

        self.in_channels = in_channels
        out_channels = 3

        #BackBone Blocks
        self.conv1 = nn.Sequential(nn.Conv2d(3, in_channels, kernel_size=3, padding=1))
        self.conv2_x = self._make_layer(block, in_channels, num_block[0], 1)# num_block=2#
        self.conv3_x = self._make_layer(block, in_channels, num_block[1], 1)
        self.conv4_x = self._make_layer(block, in_channels, num_block[2], 1)
        self.conv5_x = self._make_layer(block, in_channels, num_block[3], 1)
        self.conv6_x = self._make_layer(block, in_channels, num_block[4], 1)

        #Reconstruction Network
        self.mainconv1 = nn.Conv2d(67, out_channels * (up_scale ** 2), (3, 3), (1, 1), (1, 1)) #pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

        # Top layer
        self.toplayer = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)#Shared Lateral Layer
        self.latlayer4 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)#Not Used


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion#update inchannel

        return nn.Sequential(*layers)


    def forward(self, x):

        # Bottom UP
        c1 = self.conv1(x)
        c2 = self.conv2_x(c1)
        c3 = self.conv3_x(c2)
        c4 = self.conv4_x(c3)
        c5 = self.conv5_x(c4)
        c6 = self.conv6_x(c5)


        # Top Down
        p5 = self.toplayer(c6)  # torch.Size([2, 256, 11, 11])

        p4 = torch.add(p5, self.latlayer1(c5))
        p3 = torch.add(p4, self.latlayer2(c4))
        p2 = torch.add(p3, self.latlayer3(c3))
        p1 = torch.add(p2, self.latlayer3(c2))

        # Smooth
        p44 = self.smooth1(p4)
        p33 = self.smooth2(p3)
        p22 = self.smooth3(p2)
        p11 = self.smooth4(p1)

        #Reconstruction Network
        x1 = p11 + p22 + p33 + p44
        x1 = torch.cat([x1, x], dim=1)
        x = self.pixel_shuffle(self.mainconv1(x1))

        return x

def resnet18(in_channels, up_scale):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2, 2], in_channels, up_scale)

def test():
    model = resnet18(32, 4)#we have this model from the depth of 16 to 128
    print(model)
    modelgp = model.cuda()
    summary(modelgp, (3, 192, 192))

#test()