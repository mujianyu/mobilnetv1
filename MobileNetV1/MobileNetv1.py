import torch.nn as nn
from DepthSeperabelConv2d import  DepthSeperabelConv2d
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

class MobileNetV1(nn.Module):

    def __init__(self, width_multiplier=1, class_num=1000):
        super().__init__()
        # 使用宽度乘数来控制网络的宽度 将alpha作为一个超参数 在所有输入通道M和输出通道N上乘以alpha
        # 这里没有Rho 因为Rho 是隐式的改变分辨率
        alpha = width_multiplier
        # 这里第一个卷积不是深度可分离卷积 而是一个3X3X3X32步长为2的卷积 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(alpha*32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(alpha*32)),
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(
            DepthSeperabelConv2d(int(alpha * 32), int(alpha * 64), 1),
            DepthSeperabelConv2d(int(alpha * 64), int(alpha * 128), 2),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 1),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 256), 2),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 1),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 512), 2),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 1024), 2),
            DepthSeperabelConv2d(int(alpha * 1024), int(alpha * 1024), 2)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(alpha * 1024), class_num)
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def speed(model, name):
    t0 = time.time()
    #input = torch.rand(1,3,224,224).cuda()
    # batch = 1 通道数为3 分辨率224*224
    input = torch.rand(1,3,224,224).cuda()    
    #Variable是将tensor封装了下 用于自动求导 可以通过data获取tensor 相当于torch.no_grad()
    # input = Variable(input, volatile = True)
    with torch.no_grad():
        input = input
        t1 = time.time()
        model(input)
        t2 = time.time()
        #进行100次迭代
        for i in range(100):
            model(input)
        t3 = time.time()
        
        torch.save(model.state_dict(), "test_%s.pth"%name)
        print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    vgg16 = models.vgg16(num_classes=2).cuda()
    mobilenet = MobileNetV1().cuda()
    speed(vgg16, 'vgg16')
    speed(mobilenet, 'mobilenet')