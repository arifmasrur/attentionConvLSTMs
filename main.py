from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle

from utils.dataset import WildFireDataset
from utils.dataset import WildFireDataset_Test
from utils.dataset import WildFireDataset_valid
from networks.ConvLSTM import ConvLSTM
#from networks.ConvLSTM_CSA import ConvLSTM
#from networks.ConvLSTM_SCA import ConvLSTM
#from networks.SAN_ConvLSTM import ConvLSTM

import torch
import torch.nn as nn
from torchinfo import summary


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



trainDataSetDir = '/content/drive/MyDrive/Data/Train'
testDataSetDir = '/content/drive/MyDrive/Data/Test'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3x3_16_3x3_32_3x3_64') 
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    name = args.config
    if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64 import config
    elif name == '3x3_32_3x3_64_3x3_128': from configs.config_3x3_32_3x3_64_3x3_128 import config
    
    name = "Errors_ConvLSTM"
    logger = build_logging(config)
    
    model = ConvLSTM(config).to(config.device)
    summary(model)


    criterion = torch.nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
    

    test_dataset = WildFireDataset_Test(config, logger, testDataSetDir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
                            
                            
    patience = 10
    min_val_loss = 9999
    epoch_interval = 1
    counter = 0

    train_records, val_records, tst_records = [], [], []
    test_loss_fire, test_loss_Nofire, pred_acc = [], [], []
    test_losses10T, test_fire_losses10T, test_nofire_losses10T = [], [], []
    
    losses_all_fire_nofire_10T = {}

    for epoch in range(config.epochs):
        epoch_records = train(config, logger, epoch, model, train_loader, criterion, optimizer)
        epoch_records = np.mean(epoch_records['loss'])
        train_records.append(epoch_records)
        

        test_records, test_records_fire, test_records_Nofire, accuracy_class, test_losses10, test_fire_losses10, test_nofire_losses10 = test(config, logger, epoch, model, test_loader, criterion)
        test_records = np.mean(test_records['loss'])
        tst_records.append(test_records)
        
        test_records_fire = np.mean(test_records_fire['loss'])
        test_loss_fire.append(test_records_fire)
        
        test_records_Nofire = np.mean(test_records_Nofire['loss'])
        test_loss_Nofire.append(test_records_Nofire)
        
        pred_accuracy = np.mean(accuracy_class['class_accuracy'])
        print("Prediction Accuracy List ---------------------->>>>> ", accuracy_class)
        print("Prediction Accuracy---------------------->>>>> ", pred_accuracy)
        
        pred_acc.append(pred_accuracy)
        
        
        test_losses10T.append(test_losses10) 
        test_fire_losses10T.append(test_fire_losses10)
        test_nofire_losses10T.append(test_nofire_losses10)
        
        
        #if epoch%5==0:
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Non-attention_originalConfig_trained_model_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Non-attention_trained_model_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/ConvLSTM_CSA_model_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Pair_SAN_trained_model_FIRE_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Patch_SAN_trained_FIRE_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/NAconvlstm_trained_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/SCA_convlstm_trained_%d.pth"%epoch)
        
        if min_val_loss > test_records**0.5:
          min_val_loss = test_records**0.5
          print("Saving...")

          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_ALL_SCAconvlstm_corrected.pth")
          torch.save(model.state_dict(), "/content/drive/MyDrive/model_outputs/NAConvLSTM_FINAL.pth")
          counter = 0
        else:
          counter += 1
          
        if counter == patience:
          break
          
        #if epoch % epoch_interval == 0:
        #  print("Epoch: %d, train_loss: %1.15f, test_loss: %1.15f" % (epoch, epoch_records, test_records), "lr: ", optimizer.param_groups[0]["lr"])

        #if epoch%5==0:
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Non-attention_originalConfig_trained_model_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Non-attention_trained_model_FIRE_lr04_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/ConvLSTM_CSA_model_FIRE_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/ConvLSTM_SCA_model_FIRE_%d.pth"%epoch)
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Pair_SAN_trained_model_%d.pth"%epoch)

        
        plt.plot(range(epoch + 1), train_records, label='train')
        #plt.plot(range(epoch + 1), val_records, label='valid')
        plt.plot(range(epoch + 1), tst_records, label='test')
        plt.legend()
        #plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.savefig(os.path.join('/content/drive/MyDrive', 'model_outputs',  '{}.png'.format(name)))
        plt.close()
        
    print("@@@@@@@@@___//\\____@@@@@@@@@", train_records, tst_records, test_loss_fire, test_loss_Nofire, pred_acc)
   
        
    ## Save 10T losses
    losses_all_fire_nofire_10T["all_losses10T"] = test_losses10T
    losses_all_fire_nofire_10T["fire_losses10T"] = test_fire_losses10T
    losses_all_fire_nofire_10T["nofire_losses10T"] = test_nofire_losses10T
    
    out = open(os.path.join(r"/content/drive/MyDrive/model_outputs", "NAConvLSTM_losses_10T.pkl") ,'wb')
    pickle.dump(losses_all_fire_nofire_10T, out)
    out.close()
    
    
    
    

if __name__ == '__main__':
    main()
