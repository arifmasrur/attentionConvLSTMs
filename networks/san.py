import torch
import math
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from networks.sa import functional as F1
from networks.sa import functions
#from lib.sa.modules import Subtraction, Subtraction2, Aggregation

torch.backends.cudnn.benchmark = False

####### Modules #######

class Aggregation(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input, weight):
        return F1.aggregation(input, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
        
class Subtraction(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input):
        return F1.subtraction(input, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
        
        
class Subtraction2(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction2, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input1, input2):
        return F1.subtraction2(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
                


###################### -----------------S A M------------------- ######################       



def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc



class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes,  share_planes=4, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1) # (35, 17, 110, 110)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1) # (35, 17, 110, 110)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1) # (35, 112, 110, 110)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(7, 2) + 1)), 
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(7, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), 
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(7, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(7, dilation=2, padding=4, stride=stride)
            self.pad = nn.ReflectionPad2d(2)
        self.aggregation = Aggregation(7, stride, (dilation * (kernel_size- 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        ## Query_X = do a mask on X1's (*** if X1 is query ***) selected fire and neighboring pixels [on the first layer]
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        #print(x1.shape, x2.shape, x3.shape)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])  ## (B, rel_planes, 1, 12100) 
            
            x2_unflod = self.unfold_j(self.pad(x2)) ## (B, [rel_planes * {K*K}], 12100), e.g. (2, [2 * {3*3}], 12100) => (2, 18, 12100); 
            # Block size = 3 x 3 = 9, thus each block contains 9 spatial locations each containing a C-channeled (i.e. in this case 2) vector 
            #print(x2_unflod.shape)
            
            x2 = x2_unflod.view(x.shape[0], -1, 1, x1.shape[-1]) ## (B, 18, 1, 12100) <== just added extra dimesnion (as 3rd) to match x1's shape, because they need to be concatenated next. 
            #print("x2 unflod shape: ", x2_unflod.shape)
            w_convW = self.conv_w(torch.cat([x1, x2], 1))  ## (B, 36, 1, 12100)
            #print("W conv shape: ", w_convW.shape)
            
            #w = w_convW.view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1]) ## (B, 4, 9, 12100) <== here preserved 3rd dimesnion as K x K = 9. Why?
            w = w_convW.view(x.shape[0], -1, pow(7, 2), x1.shape[-1]) 
            

            #print("x, x1, x2, x3, W shape: ", x.shape, x1.shape, x2.shape, x3.shape, w.shape)
            
            
        x = self.aggregation(x3, w)
        #print("******************** ^^ __ ^^ *********************", x.shape)
        return x
