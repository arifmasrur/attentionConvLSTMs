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
    if i <= 1975:
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
        outputs = torch.from_numpy(pkl_data[1]).permute(0, 3, 1, 2).contiguous()
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







from configs.config_3x3_16_3x3_32_3x3_64 import config
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions import train
from utils.functions import valid
from utils.functions import test

trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train'
testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test'

train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True)


test_dataset = WildFireDataset_Test(config, logger, testDataSetDir, split='test')
test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, num_workers=config.num_workers, shuffle=False, pin_memory=True)

print(type(train_dataset))