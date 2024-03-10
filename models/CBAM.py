# Channel and Spatial Attention Module
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    # features has shape [batch_size, channels, height, width]
    def __init__(self, features, reduction):
        super(ChannelAttention, self).__init__()

        ''' CBAM: CHANNEL ATTENTION MODULE '''
        self.global_max_pool = nn.AdaptiveMaxPool2d()
        self.global_avg_pool = nn.AdaptiveAvgPool2d()

        # sigmoid(MLP(avgPool(F)) + MLP(maxPool(F)))
        self.fc1 = nn.Linear(features, features // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(features // reduction, features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x).view(x.size(0), -1))
        avg_out = self.fc2(self.relu(avg_out)).view(x.size(0), x.size(1), 1, 1)

        max_out = self.fc1(self.max_pool(x).view(x.size(0), -1))
        max_out = self.fc2(self.relu(max_out)).view(x.size(0), x.size(1), 1, 1)

        out = self.sigmoid(avg_out + max_out)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bias=False):
        super(SpatialAttention, self).__init__()

        ''' CBAM: SPATIAL ATTENTION MODULE '''
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # two feature maps from max pool and avg pool are spliced and integrated
        # then dimension is reduced by convolution operation
        # feature map MS generated after normalization by sigmoid
        # below, f is convolution operation
        # sigmoid(f^(kxk([avgPool(F); MaxPool(F)]))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, features, kernel_size=7, spatial=False):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(features)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size) if spatial else None

    def forward(self, x):
        x = self.channel_attention(x)
        if self.spatial_attention is not None:
            x = x * self.spatial_attention(x)
        return x