import torch.nn as nn
import torch.nn.functional as F

class LightCaptchaModel(nn.Module):
    def __init__(self, config):
        super(LightCaptchaModel, self).__init__()
        
        # 减少通道数
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 使用全局平均池化替代大量参数的全连接层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 轻量级全连接层
        self.fc = nn.Linear(128, config['captcha']['length'] * len(config['captcha']['charset']))
        
        self.captcha_length = config['captcha']['length']
        self.num_classes = len(config['captcha']['charset'])
        
    def forward(self, x):
        # 使用ReLU6限制激活范围，有利于定点量化
        x = F.relu6(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu6(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu6(self.conv3(x))
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x.view(-1, self.captcha_length, self.num_classes) 