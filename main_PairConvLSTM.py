# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle

from utils.dataset import WildFireDataset
from utils.dataset import WildFireDataset_Test
from utils.dataset import WildFireDataset_valid

from networks.Pair_ConvLSTM import ConvLSTM

import torch
import torch.nn as nn
from torchinfo import summary


from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions_attentions import train
from utils.functions_attentions import valid
from utils.functions_attentions import test
#from networks.CrossEntropyLoss import CrossEntropyLoss


from networks.BinaryDiceLoss import BinaryDiceLoss
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#trainDataSetDir = '/content/drive/My Drive/Wildfire/fire_train'
#testDataSetDir = '/content/drive/My Drive/Wildfire/fire_test'

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test'

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_corrected'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test_corrected'

trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_corrected_New1'
testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Test_corrected_New'

#trainDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_Test_NEW1/Train'
#testDataSetDir = '/home/arifm/usda/ConvLSTM.pytorch/Data_ConvLSTM/Train_Test_NEW1/realistic_test'



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3x3_16_3x3_32_3x3_64') 
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    name = args.config
    #if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64_original import config
    if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64 import config
    elif name == '3x3_32_3x3_64_3x3_128': from configs.config_3x3_32_3x3_64_3x3_128 import config
    logger = build_logging(config)
    
    model = ConvLSTM(config).to(config.device)
    summary(model)
    #model = ConvLSTM(config)
    #model = nn.DataParallel(model,device_ids=config.device)
    #model.cuda()
    
    #criterion = CrossEntropyLoss().to(config.device)
    criterion = torch.nn.MSELoss().to(config.device)
    #criterion = BinaryDiceLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = WildFireDataset(config, logger, trainDataSetDir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
    
    #valid_dataset = WildFireDataset_valid(config, split='valid')
    #valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
    #                        num_workers=config.num_workers, shuffle=False, pin_memory=True)
                            
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
        #print("@@@@@@@@@@@@@@@@@@@@@@@     E P O C H    @@@@@@@@@@@@@@@@@@@@@@@", epoch)
        epoch_records = train(config, logger, epoch, model, train_loader, criterion, optimizer)
        epoch_records = np.mean(epoch_records['loss'])
        train_records.append(epoch_records)
        #print("@@@@@@@@@___//\\____@@@@@@@@@", train_records, tst_records)
        #valid_records = valid(config, logger, epoch, model, valid_loader, criterion)
        #val_records.append(np.mean(valid_records['loss']))

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
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Patch_SAN_trained_%d.pth"%epoch)
        
        if min_val_loss > test_records**0.5:
          min_val_loss = test_records**0.5
          print("Saving...")
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_patch.pth")
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_original_ALL_patch.pth")
          #torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/best_model_varwise_ALL_pair.pth")
          torch.save(model.state_dict(), "/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM/Corrected_model_PAIRConvLSTM_FINAL.pth")
          counter = 0
          
        else:
          counter += 1
          
        if counter == patience:
          break
          
        #if epoch % epoch_interval == 0:
        #  print("Epoch: %d, train_loss: %1.15f, test_loss: %1.15f" % (epoch, epoch_records, test_records), "lr: ", optimizer.param_groups[0]["lr"])

        
        plt.plot(range(epoch + 1), train_records, label='train')
        #plt.plot(range(epoch + 1), val_records, label='valid')
        plt.plot(range(epoch + 1), tst_records, label='test')
        plt.legend()
        #plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.savefig(os.path.join('/home/arifm/usda/ConvLSTM.pytorch', 'Output_ConvLSTM',  '{}.png'.format(name)))
        plt.close()
        
        
    print("@@@@@@@@@___//\\____@@@@@@@@@", train_records, tst_records, test_loss_fire, test_loss_Nofire, pred_acc)
    
    ## Save 10T losses
    losses_all_fire_nofire_10T["all_losses10T"] = test_losses10T
    losses_all_fire_nofire_10T["fire_losses10T"] = test_fire_losses10T
    losses_all_fire_nofire_10T["nofire_losses10T"] = test_nofire_losses10T
    
    out = open(os.path.join(r"/home/arifm/usda/ConvLSTM.pytorch/Output_ConvLSTM", "Pair_ConvLSTM_losses_10T_FINAL.pkl") ,'wb')
    pickle.dump(losses_all_fire_nofire_10T, out)
    out.close()
    

if __name__ == '__main__':
    main()
    
    
    
    
    



