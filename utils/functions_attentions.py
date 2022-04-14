# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


import pickle

from configs.config_3x3_16_3x3_32_3x3_64 import config


def subArray_fire(x, y):
    ## --------------- Target ---------------
    x = x.permute(1, 2, 0, 3, 4).contiguous()
    #print("--------------- subarray fire shape ", x.shape)
    y = y.permute(1, 2, 0, 3, 4).contiguous()
    
    target_10s = []
    output_10s = []
    
    for t in range(10):
        #print(x.shape)
        x1 = x[t]
        #x = torch.flatten(x).cpu().detach().numpy()
        x2 = torch.flatten(x1)
    
        ## Get index of Target values > 0 [[i.e. Fire cells only]]
        #label_grater_than_0 = np.argwhere(x > 0)
        mask = x2 > 0
        indices = torch.nonzero(mask)
    
        ## --------------- Subset target cells based on indices ---------------
        xx = x2[indices].squeeze()
        
        ## --------------- Subset output cells based on fire cell index ---------------
        y1 = y[t]
        y2 = torch.flatten(y1)
        yy = y2[indices].squeeze()
        xx_yy = torch.stack([xx, yy], dim=0)
        #print(xx_yy.shape)
       
        target_10s.append(xx_yy[0].cpu().detach().numpy())
        output_10s.append(xx_yy[1].cpu().detach().numpy())
    
    target_10s_mean = torch.from_numpy(np.asarray(np.mean(np.hstack(target_10s))))
    output_10s_mean = torch.from_numpy(np.asarray(np.mean(np.hstack(output_10s))))
    
    #print(target_10s[0].shape, output_10s[0].shape)
    #print(target_10s_mean, output_10s_mean)
    
    target_output_10s = [target_10s, output_10s]

    return target_10s_mean, output_10s_mean, target_output_10s
          

def subArray_Nofire(x, y):
    ## --------------- Target ---------------
    x = x.permute(1, 2, 0, 3, 4).contiguous()
    #print("--------------- subarray fire shape ", x.shape)
    y = y.permute(1, 2, 0, 3, 4).contiguous()
    
    target_10s = []
    output_10s = []
    
    for t in range(10):
        #print(x.shape)
        x1 = x[t]
        #x = torch.flatten(x).cpu().detach().numpy()
        x2 = torch.flatten(x1)
    
        ## Get index of Target values > 0 [[i.e. Fire cells only]]
        #label_grater_than_0 = np.argwhere(x > 0)
        mask = x2 == 0
        indices = torch.nonzero(mask)
    
        ## --------------- Subset target cells based on indices ---------------
        xx = x2[indices].squeeze()
        
        ## --------------- Subset output cells based on fire cell index ---------------
        y1 = y[t]
        y2 = torch.flatten(y1)
        yy = y2[indices].squeeze()
        xx_yy = torch.stack([xx, yy], dim=0)
        #print(xx_yy.shape)
       
        target_10s.append(xx_yy[0].cpu().detach().numpy())
        output_10s.append(xx_yy[1].cpu().detach().numpy())
    
    target_10s_mean = torch.from_numpy(np.asarray(np.mean(np.hstack(target_10s))))
    output_10s_mean = torch.from_numpy(np.asarray(np.mean(np.hstack(output_10s))))
    
    #print(target_10s[0].shape, output_10s[0].shape)
    #print(target_10s_mean, output_10s_mean)
    
    target_output_10s = [target_10s, output_10s]

    return target_10s_mean, output_10s_mean, target_output_10s
    



def predClasses(targets, outputs):
            
    ### Create confusion matrix to get classification error 
    
    targets_reshaped = targets.permute(1, 2, 0, 3, 4).contiguous()
    class_targets = targets_reshaped[0][0]
    targets_class = torch.flatten(torch.where(class_targets > 0, 1, 0))
    
    mask1 = targets_class > 0
    indices1 = torch.nonzero(mask1)
    xx = targets_class[indices1].squeeze()
    
    outputs_reshaped = outputs.permute(1, 2, 0, 3, 4).contiguous()
    class_outputs = outputs_reshaped[0][0]     
    y = torch.flatten(class_outputs)
    yy = y[indices1].squeeze()
    
    #outputs_reshaped = outputs.permute(1, 2, 0, 3, 4).contiguous()
    #class_outputs = outputs_reshaped[0][0]        
    
    
    outputs_class = torch.where(yy > 0, 1, 0)
  
    #yy = outputs_class[indices2].squeeze()
    #print("Target Fire: ", len(xx), " Predicted Fire: ", len(yy))
    
    classes = torch.stack([xx, outputs_class], dim=0)
    
    return classes
    
    
    

def train(config, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_records = {'loss': []}
    num_batchs = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print(inputs.max())

        ## Make sure inputs and targets are calcluated over 110 x 100 regions.
        inputs = inputs.float().to(config.device)
        targets = targets.float().to(config.device)
        #outputs = model(inputs)
        outputs, attention = model(inputs)
        #print("----------------------------targets & outputs shape: ", targets.shape, outputs.shape)
        
        #targets_sub = subArray(targets, outputs)[0].to(config.device)
        #outputs_sub = subArray(targets, outputs)[1].to(config.device)
        
        #print(type(outputs), type(outputs_sub))
            
        losses = criterion(outputs, targets).to(config.device)
        #losses = criterion(outputs_sub, targets_sub)
        #print("LOSS ************************* Epoch {}\tBI:{}:" .format(epoch, batch_idx, losses))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_records['loss'].append(losses.item())
        
        if batch_idx and batch_idx % config.display == 0:
            logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records

def valid(config, logger, epoch, model, valid_loader, criterion):
    model.eval()
    epoch_records = {'loss': []}
    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            ## Make sure inputs and targets are calcluated over 110 x 100 regions.
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs, attention = model(inputs)
            losses = criterion(outputs, targets)
            epoch_records['loss'].append(losses.item())
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[V] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records
    
    

def test(config, logger, epoch, model, test_loader, criterion):
    model.eval()
    epoch_records = {'loss': []}
    test_loss_Nofire = {'loss': []}
    test_loss_fire = {'loss': []}
    accuracy = {'class_accuracy': []}
    
    
    ## Losses at 10 time-steps
    epoch_losses10T = {'loss10T': []}
    fire_losses10T = {'loss10T': []}
    nofire_losses10T = {'loss10T': []}
    
    num_batchs = len(test_loader)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            ## Make sure inputs and targets are calcluated over 110 x 100 regions.
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            #outputs = model(inputs)
            outputs, attention = model(inputs)
            
            
            #pickle.dump(attention, open(os.path.join(config.attention_dir, "attention.pkl"),'wb'))
                        
            ### Fire and NoFire-only losses
            losses = criterion(outputs, targets).to(config.device)
            #print("Loss: ", losses)
            
            targets_reshaped = targets.permute(1, 2, 0, 3, 4).contiguous()
            outputs_reshaped = outputs.permute(1, 2, 0, 3, 4).contiguous() 
            losses_10T = []           
            for ts in range(10):
                target_t = targets_reshaped[ts]
                output_t = outputs_reshaped[ts]

                losses_t = criterion(output_t, target_t).to(config.device)
                #print(losses_t)
                losses_10T.append(losses_t)
            epoch_losses10T['loss10T'].append(losses_10T)
                
            
            ### Fire-only losses
            target_fire, output_fire, fire_target_output_10t = subArray_fire(targets, outputs)
            targets_fire = target_fire.to(config.device)
            outputs_fire = output_fire.to(config.device)
            
            losses_fire = criterion(outputs_fire, targets_fire).to(config.device)
            
            #print("+++++++++++++++++++++++++++++++targets_fire & outputs_fire shape: ", targets_fire.shape, outputs_fire.shape)
            
            ##-- losses at 10 time steps
            fire_losses_10T = []           
            for ts in range(10):
                
                target_t1 = torch.from_numpy(fire_target_output_10t[0][ts])
                output_t1 = torch.from_numpy(fire_target_output_10t[1][ts])
                #print(output_t, target_t)

                losses_t1 = criterion(output_t1, target_t1).to(config.device)
                #print(losses_t)
                fire_losses_10T.append(losses_t1)
                
            fire_losses10T['loss10T'].append(fire_losses_10T)            
            
            
            
            ### NoFire-only losses
            target_Nofire, output_Nofire, noFire_target_output_10t = subArray_Nofire(targets, outputs)
            targets_Nofire = target_Nofire.to(config.device)
            outputs_Nofire = output_Nofire.to(config.device) 
          

            losses_Nonfire = criterion(outputs_Nofire, targets_Nofire).to(config.device)
            
            #-- losses at 10 time steps
            nofire_losses_10T = []           
            for ts in range(10):
                
                target_t2 = torch.from_numpy(noFire_target_output_10t[0][ts])
                output_t2 = torch.from_numpy(noFire_target_output_10t[1][ts])
                #print(output_t, target_t)

                losses_t2 = criterion(output_t2, target_t2).to(config.device)
                #print(losses_t)
                nofire_losses_10T.append(losses_t2)
                
            nofire_losses10T['loss10T'].append(nofire_losses_10T)             
            
            
            ## Classification error
     
            targets_class = predClasses(targets, outputs)[0].to(config.device)
            outputs_class = predClasses(targets, outputs)[1].to(config.device)
            
            #print("Target Fire: ", len(targets_class), " Predicted Fire: ", len(outputs_class))
            
            accuracy_pred = accuracy_score(targets_class.detach().cpu().numpy(), outputs_class.detach().cpu().numpy())
            
            accuracy_pred = torch.from_numpy(np.array(accuracy_pred)).to(config.device)
            #print("ACCURACY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", accuracy_pred)
            #accuracy['class_accuracy'].append(accuracy_pred)
            accuracy['class_accuracy'].append(accuracy_pred.item())
           
            
            conf_mat = confusion_matrix(targets_class.detach().cpu().numpy(), outputs_class.detach().cpu().numpy())
            target_names = ['Fire', 'No-Fire']
            #print(classification_report(targets_class.detach().cpu().numpy(), outputs_class.detach().cpu().numpy(), target_names=target_names))
            
                       
            
            #losses_ssim = 1 - criterion(inputs, outputs)
            epoch_records['loss'].append(losses.item())
            test_loss_Nofire['loss'].append(losses_Nonfire.item())
            test_loss_fire['loss'].append(losses_fire.item())
            

                     
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[T] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
            if batch_idx and batch_idx % config.draw == 0:
                _, axarr = plt.subplots(2, targets.shape[1],
                                        figsize=(targets.shape[1] * 5, 10))
                for t in range(targets.shape[1]):
                    axarr[0][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap='gray')
                    axarr[1][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap='gray')
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_Pair_SAN_ConvLSTM_FIRE.png'.format(epoch, batch_idx)))
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_Pair_Var_ConvLSTM_ALL.png'.format(epoch, batch_idx)))
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_originalConfig_ConvLSTM.png'.format(epoch, batch_idx)))
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_non_attention_ConvLSTM_FIRE_lr04.png'.format(epoch, batch_idx)))
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_ConvLSTM_CSA_FIRE.png'.format(epoch, batch_idx)))
                #plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_ConvLSTM_SCA_FIRE.png'.format(epoch, batch_idx)))
                
                plt.close()
    return epoch_records, test_loss_fire, test_loss_Nofire, accuracy, epoch_losses10T, fire_losses10T, nofire_losses10T