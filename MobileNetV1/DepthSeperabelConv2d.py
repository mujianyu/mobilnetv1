import torch.nn as nn
# define DepthSeperabelConv2d with depthwise+pointwise
class DepthSeperabelConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            # 3是核的大小  1是padding groups分成groups个组 这里将输入通道分成input_channels组  bias=False不使用偏置
            nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)#Relu6 max(0,6)
        )#deepwise conv BN ReLU
        self.pointwise = nn.Sequential(
            # 1是核的大小   这里将输入通道分成input_channels组  bias=False不使用偏置
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )#pointwise conv BN ReLU
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

