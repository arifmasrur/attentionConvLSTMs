# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from networks.cbam import CBAM
#from networks.cbam_copy import *
from networks.cbam import *


import torch
import torch.nn as nn


class CBAM(nn.Module):
    #def __init__(self, in_channels, num_features, kernel_size, padding, stride, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
    def __init__(self, in_channels, num_features, kernel_size, padding, stride, reduction_ratio=4, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(in_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(in_channels, num_features, kernel_size, padding, stride)
    def forward(self, x):
        #print("----- Channel Gate's input shape: ", x.shape)
        x_out = self.ChannelGate(x)
        #print("========================== ", x_out.shape)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
            #print("++++++++++++++++++++++ ", x_out.shape)
        return x_out
        

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        #print(out_channels)
        return nn.Sequential(CBAM(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False))
            

    def forward(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)

            #print("Combined's shape: {}".format(combined.shape))
            gates = self.conv(combined)
            #print("gates' shape -------- ", gates.shape)
            #print(self.num_features)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            #print(ingate.shape)
            
                       
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        finalOutputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)
        #print("Output's shape: {}".format(finalOutputs.shape))

        return finalOutputs



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            #layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            if 'convlstm' in layer:
                x = getattr(self, layer)(x)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~convlstm~~~~~~~~~~~~~~~~~~~~~~", x.shape)
            else:
                x = getattr(self, layer)(x)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~conv~~~~~~~~~~~~~~~~~~~~~~~", x.shape)
            #print(x.shape)
            if 'conv_' in layer: 
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
                outputs.append(x)
                
                
            if 'convlstm' in layer: outputs.append(x)
            #print("Done...encoder")
        return outputs

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            #layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == 'sigmoid': layers.append(nn.Sigmoid())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            #layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer:
                x = encoder_outputs[idx]
                #print("** deconv INPUT " + str(x.shape))
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
                #print("** deconv OUTPUT " + str(x.shape))
            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                #print("## convlstm INPUT " + str(x.shape))
                x = getattr(self, layer)(x)
                #print("## convlstm OUTPUT " + str(x.shape))
                encoder_outputs[idx] = x
                #print("Done decoder")
        return x


class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    from thop import profile
    from configs.config_3x3_16_3x3_32_3x3_64 import config
    model = ConvLSTM(config)
    flops, params = profile(model, inputs=(torch.Tensor(2, 30, 7, 110, 110),))
    print(flops / 1e9, params / 1e6)

