import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, num_feature, kernel_size, padding, stride, bn=True, bias=False):
        super().__init__()
        #self.out_channels = self.num_feature
        self.conv = nn.Conv2d(in_channels, num_feature,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        #self.bn = nn.BatchNorm2d(num_feature)
        #self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        #print("---------****-------- ", x.shape)
        x = self.conv(x)
        #print("----------------- ", x.shape)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        
        print("GATE Channel Size: ", self.gate_channels)
        
        self.fc1 = nn.Linear(gate_channels, gate_channels // reduction_ratio)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(gate_channels // reduction_ratio, gate_channels)
        
        #self.mlp = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #    nn.ReLU(),
        #    nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #    )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                #print("xxxxxxxxxxx avg_pool xxxxxxxxxxxxxxxx ", avg_pool.shape)
                #channel_att_raw = self.mlp( avg_pool )
                avg_pool = avg_pool.view(avg_pool.size(0), -1)
                #print("xxxxxxxxxxx avg_pool view xxxxxxxxx ", avg_pool.shape)
                channel_att_raw = self.relu1(self.fc1(avg_pool))
                #print("xxxxxxxxxxx channel_att_raw after fc1 & RELU xxxxxxxxx ", channel_att_raw.shape)
                channel_att_raw = self.fc2(channel_att_raw)
                #print("xxxxxxxxxxx channel_att_raw after fc2 xxxxxxxxx ", channel_att_raw.shape)
                
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                #channel_att_raw = self.mlp( max_pool )
                max_pool = max_pool.view(max_pool.size(0), -1)
                channel_att_raw = self.relu1(self.fc1(max_pool))
                channel_att_raw = self.fc2(channel_att_raw)
                
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                #channel_att_raw = self.mlp( lp_pool )
                lp_pool = lp_pool.view(lp_pool.size(0), -1)
                channel_att_raw = self.relu1(self.fc1(lp_pool))
                channel_att_raw = self.fc2(channel_att_raw)
                
            elif pool_type=='lse':
                # LSE pool only
                #lse_pool = logsumexp_2d(x)
                tensor_flatten = x.view(x.size(0), x.size(1), -1)
                s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
                lse_pool = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
                
                #channel_att_raw = self.mlp( lse_pool )
                lse_pool = lse_pool.view(lse_pool.size(0), -1)
                channel_att_raw = self.relu1(self.fc1(lse_pool))
                channel_att_raw = self.fc2(channel_att_raw)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        #print("Shape of channel_att_sum: ", channel_att_sum.shape)

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        #print(x.shape, scale.shape)
        return x * scale

#def logsumexp_2d(tensor):
#    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, padding, stride, bn=True, bias=False):
        super().__init__()
        #kernel_size = 7
        #self.compress = ChannelPool()
        #self.spatial = BasicConv(2, num_features, kernel_size, stride=1, padding=(kernel_size-1) // 2)
        self.spatial = BasicConv(in_channels, num_features, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        #x_compress = self.compress(x)
        #x_out = self.spatial(x_compress)
        pooled = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1), torch.min(x,1)[0].unsqueeze(1)), dim=1)
       
        x_pooled = torch.cat((pooled, x), dim=1) 
        #print(x_pooled.shape)
        x_out = self.spatial(x_pooled)
        scale = torch.sigmoid(x_out) # broadcasting
        #print("--------------------------------------------------------------------------------------- ", scale.shape)
        return scale


       
#class BasicConv(nn.Module):
#    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#        super(BasicConv, self).__init__()
#        self.out_channels = out_planes
#        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#        self.relu = nn.ReLU() if relu else None
#
#    def forward(self, x):
#        x = self.conv(x)
#        if self.bn is not None:
#            x = self.bn(x)
#        if self.relu is not None:
#            x = self.relu(x)
#        return x
        
        
#class SpatialGate(nn.Module):
#    def __init__(self):
#        super(SpatialGate, self).__init__()
#        kernel_size = 7
#        self.compress = ChannelPool()
#        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#    def forward(self, x):
#        x_compress = self.compress(x)
#        x_out = self.spatial(x_compress)
#        scale = F.sigmoid(x_out) # broadcasting
#        return x * scale
#
#class CBAM(nn.Module):
#    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#        super(CBAM, self).__init__()
#        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#        self.no_spatial=no_spatial
#        if not no_spatial:
#            self.SpatialGate = SpatialGate()
#    def forward(self, x):
#        x_out = self.ChannelGate(x)
#        if not self.no_spatial:
#            x_out = self.SpatialGate(x_out)
#        return x_out
