from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
#%matplotlib inline

import torch
import torch.optim as optim

import captum
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import pickle

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------

import os

root_dir = os.path.join(os.getcwd(), '.')
print(root_dir)

class Config:
    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("~~~~~~~~~~~~~~~~~~~ Device: ", device)
    if torch.cuda.is_available():
        num_workers = 4 * len(gpus)
        #num_workers = 2
        train_batch_size = 2
        valid_batch_size = 2 * train_batch_size
        test_batch_size = 2 * train_batch_size
    else:
        num_workers = 0
        train_batch_size = 2
        valid_batch_size = 2 * train_batch_size
        test_batch_size = 1
    data_file = 'datas/train-images-idx3-ubyte.gz'

    num_frames_input = 10
    num_frames_output = 5
    image_size = (110, 110) 
    display = 5
    draw = 5
    #input_size = (64, 64) 

    epochs = 15
    
    

#     (type, activation, in_ch (*** Change it to 7 ***), out_ch, kernel_size, padding, stride)
#    encoder = [('convlstm', '', 7, 16, 3, 1, 1),
#              ('convlstm', '', 16, 32, 3, 1, 1),
#             ('convlstm', '', 32, 16, 3, 1, 1)]

    #decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)]
    
#    encoder = [('convlstm', '', 7, 16, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
#              ('convlstm', '', 16, 32, 3, 1, 1)] # shape = [B, 7, 32, 110, 110]
#
#    decoder = [('conv', 'sigmoid', 32, 1, 1, 0, 1)]
    
#    encoder = [('convlstm', '', 7, 16, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
#              ('convlstm', '', 16, 28, 3, 1, 1)] # shape = [B, 7, 32, 110, 110]
#
#    decoder = [('conv', 'sigmoid', 28, 1, 1, 0, 1)]    
    
    encoder = [('convlstm', '', 1, 4, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
              ('convlstm', '', 28, 4, 3, 1, 1)] # shape = [B, 7, 32, 110, 110]

    decoder = [('conv', 'sigmoid', 28, 1, 1, 0, 1)]
        
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cache_dir = os.path.join(output_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    attention_dir = os.path.join(output_dir, 'attentions')
    if not os.path.exists(attention_dir):
        os.makedirs(attention_dir)

config = Config()




########################### D A T A S E T #############################################


from torch.utils.data import Dataset
import gzip
import random
import pickle
import os

def get_file_paths(root):
  print("The train directory is {}".format(root))
  f = []
  for _,_,filename in os.walk(root):
      f.extend(filename)
  f = sorted(f)
  paths = []
  i = 1 
  for file in f:
    #if i <= 4762:
    if i <= 2000:
      filepath = paths.append(os.path.join(root,file))
      i+=1
    else: 
      break
  
  print("Total paths: " + str(len(paths)))
  
  return paths


class WildFireDataset(Dataset):
    def __init__(self, config, logger, root, split='train'):
        super().__init__()
        self.paths = get_file_paths(root)
        #print(self.paths)
        self.image_size = config.image_size
        # self.input_size = config.input_size
        logger.info('Loaded {} samples ({})'.format(self.__len__(), split))

    def __getitem__(self, index):
        
        datafile = self.paths[index]
        #print(datafile)
        #print(index)
        pkl_data = pickle.load(open(datafile,"rb"))
        #print("^^^^^^^^^^^^^^^^^^^: pkl_data shape: ", pkl_data[0][0].shape)
        inputs = torch.from_numpy(pkl_data[0]).permute(0, 3, 1, 2).contiguous()
        inputs = inputs[20:30,:,:,:]
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()
        outputs = outputs[0:5,:,:,:]
        #print("^^^^^^^^^^^^^^^^^^^^^^^^", inputs.shape, outputs.shape)
        # inputs = (inputs / 255.) > 0.
        # outputs = outputs / 255.
        
        """
        Write your own data normalization step here
        """
        
        return inputs, outputs

    def __len__(self):
        return len(self.paths)



def get_file_paths_test(root):
  print("The train directory is {}".format(root))
  f = []
  for _,_,filename in os.walk(root):
      f.extend(filename)
  f = sorted(f)
  paths = []
  i = 1 
  for file in f:
    #if i <= 1190:
    if i <= 300:
      filepath = paths.append(os.path.join(root,file))
      i+=1
    else: 
      break

  print("Total paths: " + str(len(paths)))

  return paths


class WildFireDataset_Test(Dataset):
    def __init__(self, config, logger, root, split='train'):
        super().__init__()
        self.paths = get_file_paths_test(root)
        self.image_size = config.image_size
        # self.input_size = config.input_size
        logger.info('Loaded {} samples ({})'.format(self.__len__(), split))

    def __getitem__(self, index):
        
        datafile = self.paths[index]
        print(datafile)
        pkl_data = pickle.load(open(datafile,"rb"))
        inputs = torch.from_numpy(pkl_data[0]).permute(0, 3, 1, 2).contiguous()
        #print(".....................................", inputs.shape)
        #inputs = inputs[0:5,:,:,:]
        #print("+++++++++++++++++++++++++++++++++++++", inputs.shape)
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()
        #outputs = outputs[0:5,:,:,:]

        # inputs = (inputs / 255.) > 0.
        # outputs = outputs / 255.
        
        """
        Write your own data normalization step here
        """
        
        return inputs, outputs

    def __len__(self):
        return len(self.paths)


####################################################################


import logging
import time
def build_logging(config):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(config.log_dir, time.strftime("%Y%d%m_%H%M") + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'model_best.pth.tar'))


####################################################################



logger = build_logging(config)

trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Attention_data'
testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test'

train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
test_dataset = WildFireDataset_Test(config, logger, testDataSetDir, split='test')



####################################################################

import torch
import torch.nn as nn

from networks.san import *

#
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
#        conv_out = nn.Sequential(SAM(1, in_channels, in_channels//3, out_channels, 4,  kernel_size=kernel_size, stride=stride))
#        #print("******************** ^^ __ ^^ *********************", conv_out.shape)
#        return conv_out
#        return nn.Sequential(CBAM(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False))
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
#            print("Combined's shape: {}".format(combined.shape))
#            gates = self.conv(combined)
#            gatesList.append(gates)
#            print("gates' shape -------- ", gates.shape)
#            
#            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
#            print("Ingate shape: ", ingate.shape)
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
#        print("STACKED finalOutput shape: {}".format(finalOutput.shape))
#
#        finalOutputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)
#        print("finalOutputs shape: {}".format(finalOutputs.shape))
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
        conv_out = nn.Sequential(SAM(1, in_channels, in_channels//2, out_channels, 4,  kernel_size=kernel_size, stride=stride))
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
        for c in range(self.num_variables):
            #print("------------------------------------Here------------------------------------")
            
            B, S, C, H, W = inputs.shape
            hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
            cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
            outputs = []
            gatesList = []
            
            #print("~~~~~~~~~~~~~~~~", inputs.shape)

            for t in range(S):
                #print(inputs.shape, hx.shape)
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

            finalOutputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, HT, H, W) -> (B, S, HT, H, W)
            print("F--I--N--A--L @@@@@@@@@@@ O--U--T--P--U--T's shape: {}_{}".format(c, finalOutputs.shape))
            finalOutputs7.append(finalOutputs)

            gatesList7.append(gatesList)
        stackedOutputs7 = torch.stack(finalOutputs7).permute(1, 2, 0, 3, 4, 5).contiguous() # (C, B, S, HT, H, W) -> (B, S, C, HT, H, W)
            
        # Stack the finalOutputs7 into the shape of [B, 7, HT, H, W]
        # the reshape this into [B, 7*HT, H, W] (B, S, C, H, W)

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
        n = 0
        for layer in self.layers:
            n+=1
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            
            attr_name = getattr(self, layer)
            #print(attr_name)
            x, gates = getattr(self, layer)(x)
            #print(x.shape)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
            if n == 2:
                attention = gates
            print("Done...encoder")
            
        #print(len(outputs))
        return outputs, attention

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
                #print(encoder_outputs.shape, idx)
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                #print("## convlstm INPUT " + str(x.shape))
                x = getattr(self, layer)(x)
                #print("## convlstm OUTPUT " + str(x.shape))
                encoder_outputs[idx] = x
            print("Done ******************************************************************************************************************************************* decoder")
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

####################################################################

import torch.nn as nn
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

####################################################################

from torch.utils.data import DataLoader
trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
testloader = DataLoader(test_dataset, batch_size=1,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)



####################################################################

model = ConvLSTM(config).to(config.device)
criterion = BinaryDiceLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



####################################################################

#model.load_state_dict(torch.load('/dgxhome/axm733/ConvLSTM.pytorch/Output_ConvLSTM/trained_model_0.pth', map_location='cpu'))
model.load_state_dict(torch.load('/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model.pth', map_location='cpu'))


####################################################################



def reshape1(inPkl):
    yyy = []
    
    for i in range (7):
        yy = []
        for j in range(10):
            #xx = normalize(inPkl, i, j)
            xx = inPkl[i][j]
            #xx = xx.detach().cpu().numpy().mean(axis=1)
            xx = xx.detach().cpu().numpy()
            yy.append(xx)
        yyy.append(yy)
    
    print(len(yyy), len(yyy[0]), yyy[0][0].shape)
    return yyy


def reshape2(yyy):
    zzz = []
    for x in yyy:
        v = np.stack(x, axis=0)
        print(v.shape, v.max(), v.min())
        zzz.append(v)
    return zzz
        
        
####################################################################

listAttentions = []

for batch_idx, (inputs, targets) in enumerate(testloader):
      print("<<<<<<<<<<<<< Batch ID: >>>>>>>>>>>>>>", batch_idx)
      ## Make sure inputs and targets are calcluated over 110 x 100 regions.
      inputs = inputs.float().to(config.device)
      targets = targets.float().to(config.device)
      #outputs = model(inputs)
      

      input = inputs
      input.requires_grad = True
      print(input.shape)
      
      outputs, attention = model(input.float().to(config.device))
      
      
      ##########
      
      reshaped_attn1 = reshape1(attention)
      reshaped_attn2 = reshape2(reshaped_attn1)
      
      all_attn = np.stack(reshaped_attn2, axis=0)
      print(all_attn.shape, all_attn.max(), all_attn.min())
      
      
      all_attn_avg = all_attn.mean(axis=3)
      print(all_attn_avg.shape, all_attn_avg.max(), all_attn_avg.min())
      
      listAttentions.append(all_attn_avg)

      
      ##########
      
      #mean = all_attn_avg.mean(axis=None)
      #sd = np.std(all_attn_avg, axis=None)
      
      #mean =  -0.1265519
      #sd = 0.057965375
      
      minimum = -3.7182593 #-1.0623406 #-0.8712145
      maximum = 7.4266033 #11.707237 #0.37115547
      
      #print("Mean and standard deviation of attention: ", mean, sd)
      
      #all_attn_std = (all_attn_avg - mean) / sd
      all_attn_std = (2 * (all_attn_avg - minimum)/ (maximum - minimum)) - 1
      
      #print("++++++++++++++++++++ Standardized attention shape, min, max ++++++++++++++++++++: ", all_attn_std.shape, all_attn_std.min(), all_attn_std.max())
      
      #for x,y in enumerate(attention):
      #    print(x, y[x].shape)
      
      output_list = []
      output_list.append(inputs)
      output_list.append(targets)
      output_list.append(all_attn_avg)
      #output_list.append(all_attn_std)
          
      #out_name = "STD_Patch_attentions_rev_new_S" + str(batch_idx) + ".pkl"
      #pkl_out = open(os.path.join("/home/arifm/usda/ConvLSTM.pytorch/output/attentions_new", out_name),'wb')
      #pickle.dump(output_list, pkl_out)
      
      out_name = "Patch_attentions" + str(batch_idx) + ".pkl"
      #pkl_out = open(os.path.join("/home/arifm/usda/ConvLSTM.pytorch/output/attentions_non_std", out_name),'wb')
      #pickle.dump(output_list, pkl_out)
      
      
      


allAttentions = np.stack(listAttentions, axis=0)
print(allAttentions.shape)


#mean = allAttentions.mean(axis=None)
#sd = np.std(allAttentions, axis=None)

minimum = allAttentions.min()
maximum = allAttentions.max()

print("Min and max of ALL attentions: ", minimum, maximum)

#print("Mean and standard deviation of ALL attentions: ", mean, sd)





#dataiter = iter(testloader)

#test_data = dataiter.next()
#print(len(test_data))

#test_data = dataiter.next()
#print(len(test_data))

#test_data = dataiter.next()
#print(len(test_data))

#test_data = dataiter.next()
#print(len(test_data))

#images, labels = dataiter.next()


#print(outputs.shape, len(attention))

#for x,y in enumerate(attention):
#    print(x, y[x].shape)
    
    
#out_name = "Patch_attentions_ic2.pkl"
#pkl_out = open(os.path.join("/home/arifm/usda/ConvLSTM.pytorch/output/attentions", out_name),'wb')
#.pickle.dump(attention, pkl_out)

####################################################################

#
#directory = "/home/arifm/usda/ConvLSTM.pytorch/output/attentions"
#
#for filename in os.listdir(directory):
#  path = os.path.join(directory, filename)
#  infile = open(path,'rb')
#  inPkl = pickle.load(infile)
#  infile.close()
#  
#  print(type(inPkl), len(inPkl), inPkl[0].shape)
#  print(inPkl[0].min(), inPkl[0].max())
#  
#  
  
  
  




