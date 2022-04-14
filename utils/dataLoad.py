# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
#from utils.dataset import MovingMNISTDataset
from utils.dataset import WildFireDataset

import torch
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions import train
from utils.functions import valid
from utils.functions import test

from networks.BinaryDiceLoss import BinaryDiceLoss
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

trainDataSetDir = '/datas'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3x3_16_3x3_32_3x3_64')
    args = parser.parse_args()
    return args

def checkDataLoader(config, logger, train_loader):

    num_batchs = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        logger.info("The input max value {}".format(inputs.max()))
        inputs = inputs.float().to(config.device)
        targets = targets.float().to(config.device)
        
        logger.info("The input shape is {} and target shape is {}".format(inputs.shape, targets.shape))
        logger.info("The data type is {}".format(type(inputs)))
        
        break
        
        # if batch_idx and batch_idx % config.display == 0:
            # logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                # epoch_records['loss'][-1], np.mean(epoch_records['loss'])))

def main():
    args = get_args()
    name = args.config
    if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64 import config
    elif name == '3x3_32_3x3_64_3x3_128': from configs.config_3x3_32_3x3_64_3x3_128 import config
    logger = build_logging(config)

    # train_dataset = MovingMNISTDataset(config, trainDataSetDir, split='train')
    # train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            # num_workers=config.num_workers, shuffle=True, pin_memory=True)
    
    train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
    
    checkDataLoader(config, logger, train_loader)
    

if __name__ == '__main__':
    main()