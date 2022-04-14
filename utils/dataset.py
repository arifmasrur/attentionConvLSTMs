# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset
import gzip
import random
import torch
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
    if i <= 887:
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
        #inputs = inputs[20:30,:,:,:]
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()
        #outputs = outputs[0:10,:,:,:]
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
    if i <= 270:
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


class WildFireDataset_valid(Dataset):
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
