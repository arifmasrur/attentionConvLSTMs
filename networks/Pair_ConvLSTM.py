# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


from networks.san_pair import *


#class ConvLSTMBlock(nn.Module):
#    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
#        super().__init__()
#        self.num_features = num_features
#        self.conv = self._make_layer(in_channels+num_features, num_features*4,
#                                       kernel_size, padding, stride)
#                                       
#
#    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        #conv_out = nn.Sequential(SAM(1, in_channels, in_channels//2, out_channels, 8,  kernel_size=kernel_size, stride=stride))
#        conv_out = nn.Sequential(SAM(0, in_channels, in_channels//2, out_channels, 4,  kernel_size=kernel_size, stride=stride))
#        #print("******************** ^^ __ ^^ *********************", conv_out.shape)
#        return conv_out
#        #return nn.Sequential(CBAM(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False))
#
#    def forward(self, inputs):
#        '''
#
#        :param inputs: (B, S, C, H, W)
#        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
#        :return:
#        '''
#        outputs = []
#        gatesList = []
#        B, S, C, H, W = inputs.shape
#        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
#        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
#        for t in range(S):
#            combined = torch.cat([inputs[:, t], # (B, C, H, W)
#                                  hx], dim=1)
#
#            #print("Combined's shape: {}".format(combined.shape))
#            gates = self.conv(combined)
#            gatesList.append(gates)
#            #print("gates' shape -------- ", gates.shape)
#            
#            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
#            #print("Ingate shape: ", ingate.shape)
#            ingate = torch.sigmoid(ingate)
#            forgetgate = torch.sigmoid(forgetgate)
#            outgate = torch.sigmoid(outgate)
#
#            cy = (forgetgate * cx) + (ingate * cellgate)
#            hy = outgate * torch.tanh(cy)
#            outputs.append(hy)
#            hx = hy
#            cx = cy
#            
#        finalOutput = torch.stack(outputs)
#        #print("STACKED finalOutput shape: {}".format(finalOutput.shape))
#
#        finalOutputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)
#        #print("finalOutputs shape: {}".format(finalOutputs.shape))
#
#        return finalOutputs, gatesList




class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1, num_variables=7):
        super().__init__()
        self.num_features = num_features
        self.num_variables = num_variables
        if in_channels != 1:
            in_channels = int(in_channels/self.num_variables)
            #print(in_channels)
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)
                                           

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        #conv_out = nn.Sequential(SAM(1, in_channels, in_channels//2, out_channels, 8,  kernel_size=kernel_size, stride=stride))
        conv_out = nn.Sequential(SAM(0, in_channels, in_channels//2, out_channels, 4,  kernel_size=kernel_size, stride=stride))
        #print("******************** ^^ __ ^^ *********************", conv_out.shape)
        return conv_out
        #return nn.Sequential(CBAM(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False))

    def forward(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        
        #print("******************************************************************************", inputs.shape)
        
        finalOutputs7 = []
        gatesList7 = []
        # After the for loop [B, 7, HT, H, W]
        
        cc = 0
        for c in range(self.num_variables):
            cc += 1
            #print("------------------------------------Here------------------------------------")
            
            B, S, C, H, W = inputs.shape
            hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
            cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
            outputs = []
            gatesList = []
            
            #print("~~~~~~~~~~~~~~~~", inputs.shape)

            for t in range(S):
                if inputs.shape[2] == self.num_variables:
                    #list_vars = list(reversed(range(7)))
                    #start_index = list_vars[c]
                    #end_index = start_index+1
                    start_index = c
                    end_index = c+1
                          
                elif inputs.shape[2] == hx.shape[1]*self.num_variables:
                    
                    start_index = hx.shape[1]*c
                    end_index = hx.shape[1]*(c+1)
                combined = torch.cat([inputs[:, t, start_index:end_index], # (B, C, H, W)
                                      hx], dim=1)
    
                #print("Combined's shape: {}".format(combined.shape))
                gates = self.conv(combined)
                gatesList.append(gates)
                #print("gates' shape -------- ", gates.shape)
                
                ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
                #print("Ingate shape: ", ingate.shape)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                outgate = torch.sigmoid(outgate)
    
                cy = (forgetgate * cx) + (ingate * cellgate)
                hy = outgate * torch.tanh(cy)
                outputs.append(hy)
                hx = hy
                cx = cy
                
            #print(len(outputs))
            
            #print("############################################################################## VAR no:", cc)
            
            #print(torch.stack(outputs).shape)

            finalOutputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, HT, H, W) -> (B, S, HT, H, W)
            #print("F--I--N--A--L @@@@@@@@@@@ O--U--T--P--U--T's shape: {}_{}".format(c, finalOutputs.shape))
            finalOutputs7.append(finalOutputs)

            gatesList7.append(gatesList)
        stackedOutputs7 = torch.stack(finalOutputs7).permute(1, 2, 0, 3, 4, 5).contiguous() # (C, B, S, HT, H, W) -> (B, S, C, HT, H, W)
            
        # Stack the finalOutputs7 into the shape of [B, 7, HT, H, W]
        # then reshape this into [B, 7*HT, H, W] (B, S, C, H, W)

        stackedOutputs7 = stackedOutputs7.reshape(stackedOutputs7.shape[0], stackedOutputs7.shape[1], stackedOutputs7.shape[2] * stackedOutputs7.shape[3], stackedOutputs7.shape[4], stackedOutputs7.shape[5])
        #print("stackedOutputs7 shape: {}".format(stackedOutputs7.shape))
        return stackedOutputs7, gatesList7 # gateList has to be same for each variable



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
        attentions = []
        n = 0
        for layer in self.layers:
            n+=1
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            
            attr_name = getattr(self, layer)
            #print(attr_name)
            if 'convlstm' in layer:
                x, gates = getattr(self, layer)(x)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~convlstm~~~~~~~~~~~~~~~~~~~~~~", x.shape)
            else:
                x = getattr(self, layer)(x)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~conv~~~~~~~~~~~~~~~~~~~~~~~", x.shape)
            #print(x.shape)
            if 'conv_' in layer: 
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
                outputs.append(x)
            if 'convlstm' in layer: outputs.append(x)
            if n == 2:
                attention = gates
                attentions.append(attention)
            #print("Done...encoder")
            
        #print(len(outputs))
        return outputs, attentions

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
                #print("-----------in decoder---------------", x.shape)

                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
                #print("** deconv OUTPUT " + str(x.shape))
            elif 'convlstm' in layer:
                idx -= 1
                #print(encoder_outputs.shape, idx)
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                #print("## convlstm INPUT " + str(x.shape))
                x = getattr(self, layer)(x)
                #print("## convlstm OUTPUT " + str(x.shape))
                encoder_outputs[idx] = x
            #print("Done ******************************************************************************************************************************************* decoder")
        return x

class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x, y = self.encoder(x)
        x = self.decoder(x)
        return x, y

if __name__ == '__main__':
    from thop import profile
    from configs.config_3x3_16_3x3_32_3x3_64 import config
    model = ConvLSTM(config)
    flops, params = profile(model, inputs=(torch.Tensor(2, 30, 7, 110, 110),))
    print(flops / 1e9, params / 1e6)

