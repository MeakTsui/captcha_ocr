import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CRNNModel(nn.Module):
    """CNN + RNN 混合模型"""
    def __init__(self, config):
        super(CRNNModel, self).__init__()
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
        )
        
        # 计算CNN输出的特征维度
        self.num_classes = len(config['captcha']['charset'])
        self.captcha_length = config['captcha']['length']
        self.use_se = config.get('model', {}).get('use_se', False)
        self.se_reduction = int(config.get('model', {}).get('se_reduction', 16))
        if self.use_se:
            self.se = SEBlock(512, self.se_reduction)
        
        # LSTM层
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # 全连接层
        # 将每个时间步的特征映射到类别数，输出形状为 [batch, captcha_length, num_classes]
        self.fc = nn.Linear(512, self.num_classes)
        
    def forward(self, x):
        # CNN特征提取 [batch, channels, height, width]
        conv = self.cnn(x)  # [batch, 512, height/8, width/4]
        if hasattr(self, 'se'):
            conv = self.se(conv)
        
        # 调整维度用于RNN
        batch_size = conv.size(0)
        conv = conv.permute(0, 2, 3, 1)  # [batch, height/8, width/4, channels]
        conv = conv.reshape(batch_size, -1, 512)  # [batch, sequence_length, channels]
        
        # RNN处理
        rnn_out, _ = self.rnn(conv)  # [batch, sequence_length, 2*hidden_size]
        
        
        # 将序列长度重采样为 captcha_length，使用插值避免 MPS 上的整除限制
        # 输入: [B, T, 512] -> 转置为 [B, 512, T]
        rnn_out = rnn_out.transpose(1, 2)
        # 线性插值到目标长度 [B, 512, L]
        rnn_out = F.interpolate(rnn_out, size=self.captcha_length, mode='linear', align_corners=False)
        # 转回 [B, L, 512]
        rnn_out = rnn_out.transpose(1, 2)
        
        # 全连接层，直接得到每个位置的类别分布
        output = self.fc(rnn_out)  # [batch, captcha_length, num_classes]
        
        # 直接返回 (B, L, C)
        return output

class ResNetCaptcha(nn.Module):
    """ResNet based 模型"""
    def __init__(self, config):
        super(ResNetCaptcha, self).__init__()
        
        # 从配置读取是否使用精简版（更低延迟）
        model_cfg = config.get('model', {})
        tiny = bool(model_cfg.get('tiny', True))
        # 通道与每层 block 数配置
        if tiny:
            c1, c2, c3 = 32, 64, 128
            blocks = (1, 1, 1)
        else:
            c1, c2, c3 = 64, 128, 256
            blocks = (2, 2, 2)
        
        # 初始层
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        
        # ResNet层（根据 blocks 配置）
        self.layer1 = self._make_layer(c1, c1, blocks[0])
        self.layer2 = self._make_layer(c1, c2, blocks[1], stride=2)
        self.layer3 = self._make_layer(c2, c3, blocks[2], stride=2)
        
        # 输出层
        self.num_classes = len(config['captcha']['charset'])
        self.captcha_length = config['captcha']['length']
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, self.captcha_length * self.num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.fc(x)
        return x.view(-1, self.captcha_length, self.num_classes)

class DenseNetCaptcha(nn.Module):
    """DenseNet based 模型"""
    def __init__(self, config):
        super(DenseNetCaptcha, self).__init__()
        
        # 初始卷积
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Dense块
        self.dense1 = self._make_dense_block(64, 32, 6)
        self.trans1 = self._make_transition(256, 128)
        self.dense2 = self._make_dense_block(128, 32, 12)
        self.trans2 = self._make_transition(512, 256)
        self.dense3 = self._make_dense_block(256, 32, 24)
        
        # 输出层
        self.num_classes = len(config['captcha']['charset'])
        self.captcha_length = config['captcha']['length']
        self.use_se = config.get('model', {}).get('use_se', False)
        self.se_reduction = int(config.get('model', {}).get('se_reduction', 16))
        if self.use_se:
            self.se = SEBlock(1024, self.se_reduction)
        self.bn = nn.BatchNorm2d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.captcha_length * self.num_classes)
        
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1)
            ))
        return nn.ModuleList(layers)
    
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2, stride=2)
        )
    
    def forward(self, x):
        x = self.features(x)
        
        # Dense块1
        for layer in self.dense1:
            out = layer(x)
            x = torch.cat([x, out], 1)
        x = self.trans1(x)
        
        # Dense块2
        for layer in self.dense2:
            out = layer(x)
            x = torch.cat([x, out], 1)
        x = self.trans2(x)
        
        # Dense块3
        for layer in self.dense3:
            out = layer(x)
            x = torch.cat([x, out], 1)
        if hasattr(self, 'se'):
            x = self.se(x)
        
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x.view(-1, self.captcha_length, self.num_classes)
 