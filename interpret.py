from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
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

    epochs = 15

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

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test'

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_Test_NEW1/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/for_attention_cal'

trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_corrected'
testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/for_attention_cal_new'


train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
test_dataset = WildFireDataset_Test(config, logger, testDataSetDir, split='test')



####################################################################

import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)
                                       

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
            #nn.BatchNorm2d(out_channels)
            )

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
            else:
                x = getattr(self, layer)(x)
                
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
                #print("Done ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ decoder")
            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                #print("## convlstm INPUT " + str(x.shape))
                x = getattr(self, layer)(x)
                #print("## convlstm OUTPUT " + str(x.shape))
                encoder_outputs[idx] = x
                #print("Done ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ decoder")
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
model.load_state_dict(torch.load('/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_ALL_NAconvlstm_corrected.pth', map_location='cpu'))



####################################################################



####################################################################

dataiter = iter(testloader)
images, labels = dataiter.next()
#labels = labels.unsqueeze(0)



print("------------------------------: ", labels.shape)

print("------------------------------: ", images.shape)

print(labels[0][0][0][50][50])


label_img = labels[0][0][0].cpu().detach().numpy()
label_img1 = labels[0][0].cpu().detach().numpy()
print("++++++++++++++++ :", label_img.shape)

#label_img = ((label_img - label_img.min()) * (1/(label_img.max() - label_img.min()) * 255))
#print(label_img)

label_grater_than_0 = np.argwhere(label_img > 0)
label_less_than_0 = np.argwhere(label_img <= 0)


print(label_grater_than_0.shape, len(label_grater_than_0))
print(label_less_than_0.shape, len(label_less_than_0))

#print(type(label_less_than_0))

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
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", len(labelss), len(labelss[0]))
print(labelss.shape)


label_img= Image.fromarray(label_img)
plt.clf()
label_img = sns.heatmap(label_img)
plt.pause(0.1)
label_img.figure.savefig("label_img.png")



## Let's choose a test image at index ind and apply some of our attribution algorithms on it.

ind = 0

input = images[ind].unsqueeze(0)
input.requires_grad = True
print(input.shape)

#print(input[0][0])


target=labels[ind].to(config.device)
print("Target's shape", target.shape)

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
    
    #ig_list[str(i)] = img_ig_attr_test[0][9]
        
    ig_list.append(img_ig_attr_test[0][9])
    #print(min(ig_list), max(ig_list))
    ig_list_all.append(img_ig_attr_test[0])
    key = str(i)
    #ig_list_dict[key].append(img_ig_attr_test[0])
    ig_list_dict[key] = img_ig_attr_test[0]

print(len(ig_list_dict))
print(len(ig_list_all))


#for key in ig_list_dict:
#  print(key, ig_list_dict[key].shape)
#  ig_array = np.stack((ig_list), axis=0)
#  


ig_array = np.stack((ig_list), axis=0)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ig_array min max: ", ig_array.min(), ig_array.max())
ig_mean = np.mean(ig_array, axis=0)
ig_mean = (2 * (ig_mean - ig_mean.min())/ (ig_mean.max() - ig_mean.min())) - 1
print("---------------------------- IG_mean min max: ", ig_mean.min(), ig_mean.max())

#print("*******************ig_array shape: ", ig_array.shape)
#print(ig_mean.shape)


ig_array_10T = np.stack((ig_list_all), axis=0)
ig_mean_10T = np.mean(ig_array_10T, axis=0)
print("IG_mean_10T min and max: ", ig_mean_10T.min(), ig_mean_10T.max())

import pickle
import os
workspace = os.path.dirname(os.path.abspath(__file__))
#pickle.dump(ig_mean_10T, open(os.path.join(workspace, "ig_mean_10T_NAConvLSTM_masked_corrected.pkl") ,'wb'))

pickle.dump(ig_array_10T, open(os.path.join(workspace, "ig_Grids_10T_FIRE_NAConvLSTM_NEW.pkl") ,'wb'))
pickle.dump(ig_mean_10T, open(os.path.join(workspace, "ig_GridsMean_10T_FIRE_NAConvLSTM_NEW.pkl") ,'wb'))


##### Normalize #######
#######################

#ig_mean_10T = (ig_mean_10T - ig_mean_10T.mean())/ig_mean_10T.std()

#ig_max = np.max(ig_mean_10T)
#ig_min = np.min(ig_mean_10T)
#ig_Range = ig_max - ig_min
##ig_nrm = ((ig_mean_10T - ig_min)/ig_Range - 0.5) * 2   
#ig_nrm = (ig_mean_10T - ig_min)/ig_Range
#ig_mean_10T = ig_nrm 






#ig_mean_10T = (2 * (ig_mean_10T - ig_mean_10T.min())/ (ig_mean_10T.max() - ig_mean_10T.min())) - 1
#print("*******************ig_array 10T shape: ", ig_array_10T.shape)
#print(ig_mean_10T.min(), ig_mean_10T.max())
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
##ig_mean_10T_TC = np.transpose(ig_mean_10T_TC, (1,0,2,3))
##print(ig_mean_10T_TC.shape)
#
#for i in range(0, 10, 1):
#    importances = np.mean(ig_mean_10T_TC[i], axis=(1,2))
#    timestep = "T" + str(i)
#    importances_dict[timestep] = importances
#    
##print(len(importances_dict))
#
#df = pd.DataFrame.from_dict(importances_dict, orient = 'index')

#df.to_csv("temporal_importances_NEW.csv")

#importances = np.mean(img_ig_attr_test[0][i], axis=(1,2))


## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Feature Importance +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#plt.rcParams.update({'font.size': 18})

#feature_names = ["Unburnt vegetation", "Fire front (t-1)", "Ashes", "Wind speed (h)", "Wind speed (v)", "Elevation", "Moisture"]
#
#importances = np.mean(ig_mean, axis=(1,2))
#print(importances.min(), importances.max())
#
#def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
#    print(title)
#    for i in range(len(feature_names)):
#        print(feature_names[i], ": ", '%.3f'%(importances[i]))
#    x_pos = (np.arange(len(feature_names)))
#    if plot:
#        plt.figure(figsize=(12,12))
#        plt.bar(x_pos, importances, align='center', alpha=0.5)
#        plt.xticks(x_pos, feature_names, wrap=True, rotation=45, ha = 'right')
#        plt.ylabel('Attribution weights')
#        plt.title('Average Feature Importance (No-Fire)')
#        
#        plt.savefig("importance_fire_T10.png")
#        
##visualize_importances(feature_names, np.mean(img_ig_attr_test[0][0], axis=(1,2)))
#visualize_importances(feature_names, importances)
#










#### ----------------------------------------------------- 3d isosurface -----------------------------------------------------
#### -------------------------------------------------------------------------------------------------------------------------

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import cm, colors
#from scipy import special
#from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams["figure.figsize"] = [7.50, 7.50]
#plt.rcParams["figure.autolayout"] = True
#
#fig = plt.figure()
#ax = Axes3D(fig)
#
#for i in range(1, 4, 1):
#  xx = ig_mean_10T[i][6]
#  xx1 = ((xx - xx.min()) * (1/(xx.max() - xx.min()) * 255))
#  print("Minimum and maximum values: ", xx1.min(), xx1.max())
#  print(xx1.shape)
#  #print(xx)
#  x = xx1[0]
#  #print(x.shape)
#  y = xx1[1]
#  #print(y.shape)
#  print(x.max(), y.max())
#  
#  #x = ((x - x.min()) * (1/(x.max() - x.min()) * 255))
#  #y = ((y - y.min()) * (1/(y.max() - y.min()) * 255))
#  
#  x1, y1 = np.meshgrid(x, y)
#  #h1 = np.full((x1.shape), i+1)
#  h1 = xx1
#  
#  scamap = plt.cm.ScalarMappable(cmap='inferno')
#  fcolors = scamap.to_rgba(xx1)
#  norm = colors.Normalize()
#  alpha=(10-i)/10
#  
#  
#  ax.plot_surface(x1, y1, h1, alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#  #ax.plot_surface(x1, y1, h1,  rstride=1, cstride=1, facecolors=cm.jet(norm(xx)))
#  #ax.plot_surface(x1, y1, h1,  cmap=cm.coolwarm, linewidth=0, antialiased=False)
#  
#ax.view_init(azim=60, elev=16)  
#  
#
#plt.show()
#
#fig = plt.figure()
#ax = Axes3D(fig)

#xx = label_img1[0]
#print("Minimum and maximum values: ", xx.min(), xx.max())
#x = xx[:,0]
#y = xx[:,1]
#
##x = ((x - x.min()) * (1/(x.max() - x.min()) * 255))
##y = ((y - y.min()) * (1/(y.max() - y.min()) * 255))
#
#x1, y1 = np.meshgrid(x, y)
#h1 = np.full((x1.shape), 11)
#
#scamap = plt.cm.ScalarMappable(cmap='inferno')
#fcolors = scamap.to_rgba(xx)
##alpha=(10-i)/10
#ax.plot_surface(x1, y1, h1, rstride=2, cstride=2, facecolors=fcolors, alpha=0.5, cmap='inferno', linewidth=0, antialiased=False)
#
#plt.show()
  
  

#for i in range(6, 7, 1):
#  print(i)
#  for j in range(0, 10, 1):
#
#    img_ig_arr = ig_mean_10T[j][i]
#    #print(img_ig_arr.max(), img_ig_arr.min())
#    img_ig_arr = ((img_ig_arr - img_ig_arr.min()) * (1/(img_ig_arr.max() - img_ig_arr.min()) * 255))
#    
#    img_ig = Image.fromarray(img_ig_arr)
#    #print(img_ig_arr)
#    
#    plt.clf()
#    #img_ig = sns.heatmap(img_ig, vmin=0, vmax=1)
#    img_ig = sns.heatmap(img_ig)
#    #print(img_ig)
#    plt.pause(0.1)
#    #img_ig.show()
#    #plt.imshow(img_ig)
#    img_ig.figure.savefig("ig_firefront_moist_T" + str(j) + ".png")


#############################################**********^_^**********##########################################################
### Mayavi Mayavi Mayavi Mayavi Mayavi
### https://stackoverflow.com/questions/13932150/matplotlib-wrong-overlapping-when-plotting-two-3d-surfaces-on-the-same-axes 
##############################################################################################################################

#x = np.arange(-2, 2, 0.1)
#y = np.arange(-2, 2, 0.1)

#xx = ig_mean_10T[0][6]
#xx1 = ((xx - xx.min()) * (1/(xx.max() - xx.min()) * 255))
#x = xx1[0]
#y = xx1[1]
#
#mx, my = np.meshgrid(x, y, indexing='ij')
#mz1 = xx1
#
#xx = ig_mean_10T[1][6]
#xx2 = ((xx - xx.min()) * (1/(xx.max() - xx.min()) * 255))
#x = xx2[0]
#y = xx2[1]
#mx1, my1 = np.meshgrid(x, y, indexing='ij')
#mz2 = xx2
#
#
#from mayavi import mlab
#fig = mlab.figure()

#input_mlab = np.transpose(ig_mean_10T, (1, 0, 2, 3))
#input_mlab = ((input_mlab - input_mlab.min()) * (1/(input_mlab.max() - input_mlab.min()) * 255))
#volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(input_mlab[6]), vmin=np.percentile(input_mlab[6], 20), vmax=np.percentile(input_mlab[6], 100))
#volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(input_mlab[4]))
#mlab.draw()
#mlab.view(45, 45, 17, [-2.5, -4.6, -0.3])
#fig.scene.renderer.use_depth_peeling = 1

#mlab.quiver3d(input_mlab[6], input_mlab[6], input_mlab[6])
#
#mlab.pipeline.surface(np.transpose(input_mlab[6], (1,2,0)))
#mlab.pipeline.surface(input_mlab[6][0])
#mesh = mlab.mesh(input_mlab[6][0], input_mlab[6][0], input_mlab[6][0])
#mlab.pipeline.surface(mesh)
#mlab.contour3d(np.transpose(input_mlab[6], (1,2,0)), contours=10, transparent=False, opacity=0.5)
#mlab.imshow(label_img1[0])
#xx = input_mlab[6]
#x, y, z = np.mgrid[0: xx.shape[0], 0: xx.shape[1], 0: xx.shape[2]]
#
#mlab.contour3d(x*10, y, z, xx, contours=10, transparent=False, opacity=0.5)
#mlab.outline()






#np.random.seed(12345)
#x = 4 * (np.random.random(500) - 0.5)
#y = 4 * (np.random.random(500) - 0.5)
#def f(x, y):
#    return np.add(x, y)
#z = f(x, y)
#from mayavi import mlab
#myfig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
#for i in range(3):
#  z = z + 1
#  # Visualize the points
#  pts = mlab.points3d(x, y, z, z,figure=myfig, scale_mode='none', scale_factor=1)
#  # Create and visualize the mesh
#  mesh = mlab.pipeline.delaunay2d(pts,figure=myfig)
#  surf = mlab.pipeline.surface(mesh,figure=myfig)
#  pts.remove()
#mlab.axes(figure=myfig)
#mlab.show()


#
#fig = mlab.figure(figure='moisture', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#
#scalars = np.transpose(input_mlab[6], (1,2,0))   # specifying the data array
#mlab.volume_slice(scalars, slice_index=0, plane_orientation='x_axes', figure=fig)   # crossline slice
#mlab.volume_slice(scalars, slice_index=0,  plane_orientation='y_axes', figure=fig)   # inline slice
#mlab.volume_slice(scalars, slice_index=0, plane_orientation='z_axes', figure=fig)   # depth slice
#
#mlab.axes(xlabel='X', ylabel='Y', zlabel='Time steps', nb_labels=5)                             # Add axes labels 
#
#mlab.show()
#



#def f(x, y):
#    return np.add(x, y)
#
#xx = input_mlab[6][9][0]
#yy = input_mlab[6][9][1]
#
#z = f(xx, yy)
#
#myfig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
#for i in range(0, 10, 1):
#  #x = input_mlab[6][i][0]
#  #y = input_mlab[6][i][1]
#  
#  #z = f(x, y)
#  #z = z +1 
#  
#  #pts = mlab.points3d(xx, yy, z, z, figure=myfig, scale_factor=1)
#  # Create and visualize the mesh
#  #mesh = mlab.pipeline.delaunay2d(pts,figure=myfig)
#  #mesh = mlab.pipeline.array2d_source(input_mlab[6][i])
#  mesh = mlab.imshow(input_mlab[6][i])
#  surf = mlab.pipeline.surface(mesh)
#  #pts.remove()
#
#mlab.axes(figure=myfig)
#mlab.show()














#mlab.outline()

#mlab.imshow(xx1, colormap='gist_earth')

#ax_ranges = [xx1.min(), xx1.max(), xx1.min(), xx1.max(), 0, 50]
#ax_scale = [1.0, 1.0, 1.0]
#ax_extent = ax_ranges * np.repeat(ax_scale, 2)
#
#surf3 = mlab.surf(mz1, colormap='Blues')
#surf4 = mlab.surf(mz2, colormap='Oranges')

#surf3.actor.actor.scale = ax_scale
#surf4.actor.actor.scale = ax_scale
#mlab.view(45, 45, 17, [-2.5, -4.6, -0.3])
#mlab.outline(surf3, color=(.7, .7, .7), extent=ax_extent)
#mlab.axes(surf3, color=(.7, .7, .7), extent=ax_extent,
#          ranges=ax_ranges,
#          xlabel='x', ylabel='y', zlabel='z')


#surf3.actor.property.opacity = 0.5
#surf4.actor.property.opacity = 0.5
#fig.scene.renderer.use_depth_peeling = 1

#mlab.show()



















#for i in range(0, 7, 1):
#  print(i)
#
#  img_ig_arr = img_ig_attr_test[0][9][i]
#  print(img_ig_arr.max(), img_ig_arr.min())
#  img_ig_arr = ((img_ig_arr - img_ig_arr.min()) * (1/(img_ig_arr.max() - img_ig_arr.min()) * 255))
#  
#  img_ig = Image.fromarray(img_ig_arr)
#  #print(img_ig_arr)
#  
#  plt.clf()
#  #img_ig = sns.heatmap(img_ig, vmin=0, vmax=1)
#  img_ig = sns.heatmap(img_ig)
#  #print(img_ig)
#  plt.pause(0.1)
#  #img_ig.show()
#  #plt.imshow(img_ig)
#  img_ig.figure.savefig("ig_" + str(i) + ".png")


#img_ig_attr_imp = img_ig_attr_test[0][9]
#print(img_ig_attr_imp.shape)
#
#importances = np.mean(img_ig_attr_imp, axis=(1,2))
#
#print(importances)
#print(importances.shape)
#print(type(importances))
#
#feature_names = ["Unburnt vegetation", "Tree vegetation burnt", "All vegetation burnt", "Wind speed (h)", "Wind speed (v)", "Elevation", "Moisture"]
#
#for i in range(len(feature_names)):
#    print(feature_names[i], ": ", '%.3f'%(importances[i]))
#
#x_pos = (np.arange(len(feature_names)))
#print(x_pos)
#
#
#importances_dict = {}
#
#for i in range(0, 10, 1):
#    importances = np.mean(img_ig_attr_test[0][i], axis=(1,2))
#    timestep = "T" + str(i)
#    importances_dict[timestep] = importances
#    
#print(importances_dict)
#
#df = pd.DataFrame.from_dict(importances_dict, orient = 'index')
#
#df.to_csv("temporal_importances.csv")





## Feature Importance
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
#        plt.savefig("importance_T1.png")
#        
#visualize_importances(feature_names, np.mean(img_ig_attr_test[0][0], axis=(1,2)))
#



### =========================================================================================================================================
### ------------------------------------------------------ DeepLift -------------------------------------------------------------------------
### =========================================================================================================================================


#dl = DeepLift(model)
#attr_dl = attribute_image_features(dl, input.float().to(config.device), baselines=input.float().to(config.device) * 0)
#
#print("******************: ", attr_dl.shape)
#attr_dl = attr_dl.cpu().detach().numpy()
#
#
#print(attr_dl.shape)




#x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
#print(x.shape)
#im = Image.fromarray(x)
#
#import matplotlib.pyplot as plt
#plt.imshow(x)
#plt.savefig("image")

#ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)

#attr_ig, delta = attribute_image_features(ig, input.float().to(config.device), baselines=input.float().to(config.device) * 0, return_convergence_delta=True)
#attr_ig = np.transpose(attr_ig.squeeze().gpu().detach().numpy(), (1, 2, 0))
#print('Approximation delta: ', abs(delta))

