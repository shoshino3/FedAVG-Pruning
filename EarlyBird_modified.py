from __future__ import print_function

import torch
import torch.nn as nn
from collections import OrderedDict

#from filter import *


class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]
        
    def _forward_pre_hook(self, mask):
        def pre_hook(module, input):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            
                #print("module weight shape", module.weight.shape)
                #print("mask.shape", mask.shape)
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
            else:
                raise NotImplementedError(f'Unsupported module type {type(module)} in EarlyBird pre-hook')
        return pre_hook

    def unregister_mask(self, model):
        for module in model.modules():
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()

    def register_mask(self, model, mask):
        self.unregister_mask(model)  # Clear existing hooks
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.register_forward_pre_hook(self._forward_pre_hook(mask))

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
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1) 
                index += size
                
        print("mask shape from EB class is:", mask.shape)
       
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
            #print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False
