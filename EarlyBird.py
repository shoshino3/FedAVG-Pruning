from __future__ import print_function

import torch
import torch.nn as nn
from models import vgg
import numpy as np
from utils import count_zero_weights, count_parameters
import copy

#from filter import *


class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone() ##" The slice bn[index:(index+size)] 
                ##refers to the portion of the bn tensor where |w| should be placed."
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index] #Weights below this threshold will be considered for pruning.
      

        mask = torch.zeros(total)
        index = 0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.to(device)).float().to(device)
                mask[index:(index+size)] = _mask.view(-1) 
                index += size

      
        return mask

    def put(self, mask): ##to make sure we are only storing 4 mask distances
        """
        If fewer than self.epoch_keep masks are stored, it appends the new mask to self.masks.
        If the limit is reached, it removes the oldest mask (self.masks.pop(0)) before appending the new one. 
        This ensures that only the most recent self.epoch_keep masks are kept.
        """
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        
        ##calculates distance only after epoch 4
        ##self.epoch_keep = 5 (constant throughout)
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0) # calculates the proportion 
                                                                                      ##of differing elements between mask_i and mask_j.
            return True
        else:
            return False

    def early_bird_emerge(self, model):
      
     
        mask = self.pruning(model, self.percent) ## generate the mask
        self.put(mask) ## to make sure the number of masks to be kept is equal to epoch_keep
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False
    
def actual_prune(model, pruning_ratio):
    original_parameters = count_parameters(model)
    print("Original parameters: ", original_parameters)
    target_parameters = int(original_parameters * (1-pruning_ratio))
    while count_parameters(model) > target_parameters:
        model = prune_by_ratio(model, pruning_ratio*0.5)
    
    print("Pruned model parameters: ", count_parameters(model))
    return model

def prune_by_ratio(model, pruning_ratio):
    total = 0 ##to store the total number of weights in all the bn layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    # get the threshold index for each channel
    bn = torch.zeros(total)
    index = 0 
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size        
    y, _ = torch.sort(bn)
    thre_index = int(total * pruning_ratio) #change for pruning ratio
    thre = y[thre_index]
    
    cfg = [] ##number of remaining channels
    cfg_mask = [] ##mask for each layer
    
    pruned = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #pruning starts
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone() # copy the absolute values of the weight
            mask = weight_copy.gt(thre.to(device)).float().to(device) # entering 1 (if above thre) else 0
            
            # if all channels are pruned, then we will remain the channel with the largest value
            if torch.sum(mask) == 0:
                max_channel_idx = torch.argmax(weight_copy)
                mask[max_channel_idx] = 1.0
            
            pruned = pruned + mask.shape[0] - torch.sum(mask) # no. of pruned channels
            m.weight.data.mul_(mask) # put mask on the weights
            m.bias.data.mul_(mask) # put mask on the bias
            
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask))) # append the count of retained weights
            cfg_mask.append(mask.clone()) ##append the mask for each layer
            
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
            
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    #count the number of zero channels: 
    print(cfg)
    count_zero_weights(model, s = "Before removing zero channels") 
    
    # PART II: actual pruning where zeroed weights (channels) are excluded. 
    
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    
    newmodel = vgg(dataset='cifar10', cfg=cfg)

    for [m0, m1] in zip(model.modules(), newmodel.modules()): 
        
        if isinstance(m0, nn.BatchNorm2d):
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone() 
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
                
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    model = copy.deepcopy(newmodel)
    
    count_zero_weights(model, s="After removing zero channels")
        
    return model
   
    
    