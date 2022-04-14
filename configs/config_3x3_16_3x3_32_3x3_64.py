# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
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
        test_batch_size = 2 * train_batch_size
    data_file = 'datas/train-images-idx3-ubyte.gz'

    num_frames_input = 30
    num_frames_output = 30
    image_size = (110, 110) 
    display = 10
    draw = 10
    #input_size = (64, 64) 

    epochs = 15
    
    

#     (type, activation, in_ch (*** Change it to 7 ***), out_ch, kernel_size, padding, stride)
#    encoder = [('convlstm', '', 7, 16, 3, 1, 1),
#              ('convlstm', '', 16, 32, 3, 1, 1),
#             ('convlstm', '', 32, 16, 3, 1, 1)]

    #decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)]
    
    encoder = [('convlstm', '', 7, 16, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
              ('convlstm', '', 16, 32, 3, 1, 1),
              ('conv', 'relu', 32, 16, 3, 1, 1)] # shape = [B, 7, 32, 110, 110]

    decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)]



##------------------------- Other ConvLSTMs
    
#    encoder = [('convlstm', '', 7, 28, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
#              ('convlstm', '', 28, 28, 3, 1, 1),
#              ('conv', 'relu', 28, 16, 3, 1, 1)] 
#              
#
#    decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)]  
    
    
    
    

##------------------------- Variable-wise ConvLSTM

#    encoder = [('convlstm', '', 1, 4, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
#              ('convlstm', '', 28, 4, 3, 1, 1),
#              ('conv', 'relu', 28, 16, 3, 1, 1)]  
#    decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)] 






    
#    encoder = [('convlstm', '', 1, 16, 3, 1, 1), # shape = [B, 7, 16, 110, 110]
#              ('convlstm', '', 112, 16, 3, 1, 1),
#              ('conv', 'relu', 112, 64, 3, 1, 1),
#              ('conv', 'relu', 64, 32, 3, 1, 1),
#              ('conv', 'relu', 32, 16, 3, 1, 1)] 
#              
#    decoder = [('conv', 'sigmoid', 16, 1, 1, 0, 1)]
        
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
