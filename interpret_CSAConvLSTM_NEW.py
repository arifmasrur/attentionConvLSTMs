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
    if torch.cuda.is_available():
        num_workers = 4 * len(gpus)
        #num_workers = 2
        train_batch_size = 2
        valid_batch_size = 2 * train_batch_size
        test_batch_size = 1
    else:
        num_workers = 0
        train_batch_size = 1
        valid_batch_size = 2 * train_batch_size
        test_batch_size = 1
    data_file = 'datas/train-images-idx3-ubyte.gz'

    num_frames_input = 10
    num_frames_output = 10
    image_size = (110, 110) 
    display = 10
    draw = 10
    #input_size = (64, 64) 

    epochs = 20

    # (type, activation, in_ch (*** Change it to 7 ***), out_ch, kernel_size, padding, stride)
    #encoder = [('convlstm', '', 7, 16, 3, 1, 1),

    #        ('convlstm', '', 16, 32, 3, 1, 1),
 
    #         ('convlstm', '', 32, 32, 3, 1, 1)]

    #decoder = [('conv', 'sigmoid', 32, 1, 1, 0, 1)]
    
#    encoder = [('convlstm', '', 7, 16, 3, 1, 1),
#              ('convlstm', '', 16, 32, 3, 1, 1)]
#
#    decoder = [('conv', 'sigmoid', 32, 1, 1, 0, 1)]
    
    encoder = [('convlstm', '', 7, 28, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
              ('convlstm', '', 28, 28, 3, 1, 1),
              ('conv', 'relu', 28, 16, 3, 1, 1)] # shape = [B, 7, 32, 110, 110]
    decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)] 

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

config = Config()



########################################################################


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
    if i <= 10:
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
        self.image_size = config.image_size
        # self.input_size = config.input_size
        logger.info('Loaded {} samples ({})'.format(self.__len__(), split))

    def __getitem__(self, index):
        
        datafile = self.paths[index]
        pkl_data = pickle.load(open(datafile,"rb"))
        inputs = torch.from_numpy(pkl_data[0]).permute(0, 3, 1, 2).contiguous()
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()

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
    if i <= 2:
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
        print(index, datafile)
        pkl_data = pickle.load(open(datafile,"rb"))
        inputs = torch.from_numpy(pkl_data[0]).permute(0, 3, 1, 2).contiguous()
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()

        # inputs = (inputs / 255.) > 0.
        # outputs = outputs / 255.
        
        """
        Write your own data normalization step here
        """
        
        return inputs, outputs

    def __len__(self):
        return len(self.paths)


def get_file_paths_valid(root):
  print("The train directory is {}".format(root))
  f = []
  for _,_,filename in os.walk(root):
      f.extend(filename)
  f = sorted(f)
  paths = []
  i = 1 
  for file in f:
    if i <= 201:
      filepath = paths.append(os.path.join(root,file))
      i+=1
    else: 
      break

  print("Total paths: " + str(len(paths)))

  return paths




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

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_Test_NEW1/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/for_attention_cal'

trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_corrected'
testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/for_attention_cal_new'



train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
test_dataset = WildFireDataset_Test(config, logger, testDataSetDir, split='test')



####################################################################

from networks.cbam import *


import torch
import torch.nn as nn


class CBAM(nn.Module):
    #def __init__(self, in_channels, num_features, kernel_size, padding, stride, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
    def __init__(self, in_channels, num_features, kernel_size, padding, stride, reduction_ratio=8, pool_types=['avg', 'max', 'lp', 'lse'], no_spatial=False):
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
testloader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)



####################################################################

model = ConvLSTM(config).to(config.device)
criterion = BinaryDiceLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



####################################################################

#model.load_state_dict(torch.load('/dgxhome/axm733/ConvLSTM.pytorch/Output_ConvLSTM/trained_model_0.pth', map_location='cpu'))
model.load_state_dict(torch.load('/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_ALL_CSAconvlstm_corrected.pth', map_location='cpu'))



####################################################################



####################################################################

for i, (images, labels) in enumerate(testloader):
    print(i)

dataiter = iter(testloader)
images, labels = dataiter.next()
#labels = labels.unsqueeze(0)

## outputs, attention = model(images)


print("------------------------------: ", labels.shape)

print("------------------------------: ", images.shape)

print(labels[0][0][0][50][50])


label_img = labels[0][0][0].cpu().detach().numpy()
label_img1 = labels[0][0][0].cpu().detach().numpy()
print("++++++++++++++++ :", label_img.shape)

#label_img = ((label_img - label_img.min()) * (1/(label_img.max() - label_img.min()) * 255))
print(label_img)

label_grater_than_0 = np.argwhere(label_img > 0) ## For all fires and no-fire locations
label_less_than_0 = np.argwhere(label_img <= 0)

print(label_grater_than_0.shape, len(label_grater_than_0))
print(label_less_than_0.shape, len(label_less_than_0))



random.seed(11)
label_less_than_0_subset = random.sample(list(map(tuple, label_less_than_0)), 5000)
label_less_than_0_subset = np.array(label_less_than_0_subset) 

print(label_grater_than_0.shape, len(label_grater_than_0))
print(label_less_than_0.shape, len(label_less_than_0))





#### Fire and neighboring 3 x 3 areas: 
list_windows = []
for x in label_grater_than_0:
    i,j = x[0], x[1]
    
    p = [i-1, i, i+1]
    q = [j-1, j, j+1]
    
    list_index = []
    for i in p:
        for j in q:
            k = [i, j]
            list_index.append(k)
            
    list_windows.append(list_index)

    #window = fire_target[0][0][i-1:i+2, j-1:j+2]
    #print(window)

## Get only unique locations
all_index = []
for x in list_windows:
    for y in x:
        all_index.append(y)

print(len(all_index))


unique_index = set(tuple(x) for x in all_index)
print(len(unique_index))  


## Make np array 

list_indices = []

for x in unique_index:
    #print(x[0], x[1])
    i, j = x[0], x[1]
    index = [i, j]
    list_indices.append(index)


np.array(list_indices)
labelss = np.array(list_indices)











#for i in range(0, len(label_grater_than_0), 1):
#    print(label_grater_than_0[i][0], "-----", label_grater_than_0[i][1])
    


#label_img= Image.fromarray(label_img)
#plt.clf()
#label_img = sns.heatmap(label_img)
#plt.pause(0.1)
#label_img.figure.savefig("label_img.png")



## Let's choose a test image at index ind and apply some of our attribution algorithms on it.

ind = 0

input = images[ind].unsqueeze(0)
input.requires_grad = True
print(input.shape)

#print(input[0][0])


#target=labels[ind].to(config.device)
#print(target.shape)

### ************ ### ************ ### ************ ### ************ ### ************ ### ************ ###     

#def attribute_image_features(algorithm, input, **kwargs):
#    model.zero_grad()
#    tensor_attributions = algorithm.attribute(input.float().to(config.device),
#                                              target=(0, 0, 50, 50),
#                                              **kwargs
#                                             )
#    
#    return tensor_attributions


#saliency = Saliency(model)
#grads = saliency.attribute(input.float().to(config.device), target=labels[ind].item())
#grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

### =========================================================================================================================================
### ------------------------------------------------------ Integrated Gradients -------------------------------------------------------------
### =========================================================================================================================================


ig = IntegratedGradients(model)

ig_list_dict = {}
ig_list = []
ig_list_all = []


for i in range(0, len(label_grater_than_0), 1):
    #print(label_grater_than_0[i][0], "-----", label_grater_than_0[i][1])
    ig_attr_test = ig.attribute(input.float().to(config.device), baselines=input.float().to(config.device) * 0, target=(9, 0, label_grater_than_0[i][0], label_grater_than_0[i][1]), n_steps=1)
    print(ig_attr_test.shape)

    img_ig_attr_test = ig_attr_test.cpu().detach().numpy()
    print("-------------------------- img_ig_attr_test shape: ", img_ig_attr_test.shape)
    
  
    #img_ig_attr_test =  np.mean(img_ig_attr_test, axis=1)
    #print("img_ig_attr_test shape: ", img_ig_attr_test.shape)
    #ig_list.append(img_ig_attr_test)
    
    ig_list.append(img_ig_attr_test[0][9])  ## IG of the T10 time-step to predict on T11 fire/no-fire
    #print(min(ig_list), max(ig_list))
    ig_list_all.append(img_ig_attr_test[0])
    key = str(i)
    #ig_list_dict[key].append(img_ig_attr_test[0])
    ig_list_dict[key] = img_ig_attr_test[0]

print(len(ig_list_dict))
print(len(ig_list_all))



ig_array = np.stack((ig_list), axis=0)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ig_array min max: ", ig_array.min(), ig_array.max())
ig_mean = np.mean(ig_array, axis=0)
print("*******************ig_array shape: ", ig_array.shape)
print(ig_mean.shape)



### 
ig_array_10T = np.stack((ig_list_all), axis=0)
print(ig_array_10T.shape)
ig_mean_10T = np.mean(ig_array_10T, axis=0)

import pickle
import os
workspace = os.path.dirname(os.path.abspath(__file__))
pickle.dump(ig_array_10T, open(os.path.join(workspace, "ig_Grids_10T_FIRE_CSA_NEW.pkl") ,'wb'))
pickle.dump(ig_mean_10T, open(os.path.join(workspace, "ig_GridsMean_10T_FIRE_CSA_NEW.pkl") ,'wb'))

##### Normalize #######
#######################

#ig_mean_10T = (ig_mean_10T - ig_mean_10T.mean())/ig_mean_10T.std()

#ig_max = np.max(ig_mean_10T)
#ig_min = np.min(ig_mean_10T)
#ig_Range = ig_max - ig_min
##ig_nrm = ((ig_mean_10T - ig_min)/ig_Range - 0.5) * 2   
#ig_nrm = (ig_mean_10T - ig_min)/ig_Range
#
#ig_mean_10T = ig_nrm 
#print("*******************ig_array 10T shape: ", ig_array_10T.shape)
#print(ig_mean_10T.shape)
#print(ig_mean_10T[0][0].shape)


#### --------------------------------------- Temporal Change in feature importance ----------------------------------------------
#### ----------------------------------------------------------------------------------------------------------------------------



#importances_dict = {}
#
##ig_array_10T_TC = ((ig_mean_10T - ig_mean_10T.min()) * (1/(ig_mean_10T.max() - ig_mean_10T.min()) * 255))
#ig_mean_10T_TC = ig_mean_10T
#
#print(ig_mean_10T_TC.shape)
#
#
#for i in range(0, 10, 1):
#    importances = np.mean(ig_mean_10T_TC[i], axis=(1,2))
#    timestep = "T" + str(i)
#    importances_dict[timestep] = importances
#    
#print(len(importances_dict))
#
#df = pd.DataFrame.from_dict(importances_dict, orient = 'index')
#
##df.to_csv("temporal_importances_fire_CSA_ConvLSTM_NEW.csv")


## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Feature Importance +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#plt.rcParams.update({'font.size': 18})
#
#feature_names = ["Unburnt veg", "Fire front (t-1)", "Ashes", "Wind speed (h)", "Wind speed (v)", "Terrain heights", "Moisture"]
#
#importances = np.mean(ig_mean, axis=(1,2))
#
#def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
#    print(title)
#    for i in range(len(feature_names)):
#        print(feature_names[i], ": ", '%.3f'%(importances[i]))
#    x_pos = (np.arange(len(feature_names)))
#    if plot:
#        plt.figure(figsize=(12,12))
#        plt.bar(x_pos, importances, align='center')
#        plt.xticks(x_pos, feature_names, wrap=True, rotation=45, ha = 'right')
#        plt.xlabel(axis_title)
#        plt.title(title)
#        
#        #plt.savefig("CSAConvLSTM_importance_nofire_T10_NEW.png")
#        
#visualize_importances(feature_names, np.mean(img_ig_attr_test[0][0], axis=(1,2)))
#
#
#
#
#
