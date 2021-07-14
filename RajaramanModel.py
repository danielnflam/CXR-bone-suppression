#ResNet-BS model:
import torch
import torch.nn as nn
import numpy as np

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size = (3,3), stride=1, padding=1, scaling=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scaling = scaling
        
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
    def forward(self, x_in):
        x = self.act(self.conv1(x_in))
        x = self.conv2(x)
        if self.scaling is not None:
            x = x*self.scaling
        x = x + x_in
        return x


class ResNet_BS(nn.Module):
    def __init__(self, input_array_shape, num_filters=64, num_resblocks=16, res_block_scaling = 0.1):
        super().__init__()
        self.input_array_shape = input_array_shape
        self.num_filters = num_filters
        self.num_resblocks = num_resblocks
        self.res_block_scaling = res_block_scaling
        
        self.conv0 = nn.Conv2d(self.input_array_shape[-3], self.num_filters, (3,3), 1, 1)
        self.resModules = nn.ModuleList()
        for _ in range(num_resblocks):
            self.resModules.append(res_block(self.num_filters, scaling=self.res_block_scaling))
        self.conv_2ndBeforeLast = nn.Conv2d(self.num_filters, self.num_filters, (3,3), 1, 1)
        
        self.conv_output = nn.Conv2d(self.num_filters, 1, (3,3), 1, 1)
        self.sigmoid_act = nn.Sigmoid()
    def forward(self, x_in):
        x_branch = self.conv0(x_in) # residual out
        
        x_main = torch.clone(x_branch)
        for i, l in enumerate(self.resModules):
            x_main = self.resModules[i](x_main)
        x_main = self.conv_2ndBeforeLast(x_main)
        
        x_main = x_main + x_branch # residual plus main branch
        
        out = self.conv_output(x_main)
        return self.sigmoid_act(out)