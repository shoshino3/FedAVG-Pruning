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
        
    
    def _forward_pre_hooks(self, masks):
        def pre_hook(module, input):
            if isinstance(module, nn.BatchNorm2d):
                m = masks[str(module)]
                module.weight.data.mul_(m)
            else:
                raise NotImplementedError('Unsupported ' + m)
        return pre_hook

    def _unregister_mask(self, model):
            for m in model.modules():
                m._backward_hooks = OrderedDict()
                m._forward_pre_hooks = OrderedDict()

    def register_mask(self, model, masks=None):
            self._unregister_mask(model)
            layer_dict = dict()
            for layer in model.modules():
                layer_dict[str(layer)] = layer
            assert masks is not None, 'Masks should be generated first.'
            for layer_name in masks.keys():
                assert layer_name in layer_dict.keys(), 'mask key not in origin model!'
                layer_dict[layer_name].register_forward_pre_hook(self._forward_pre_hooks(masks))

    def pruning_by_mask(self, model, masks_dict):
        layer_dict = dict()
        for layer in model.modules():
            layer_dict[str(layer)] = layer
        for layer_name, mask in masks_dict.items():
            if isinstance(layer_dict[layer_name], nn.BatchNorm2d):
                layer_dict[layer_name].weight.data.mul_(mask)
    
    def get_mask_for_hooks(self, model, percent):
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
        
        masks = {}
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre).float() # Ensure the device is the same as model's weights
                masks[str(m)] = _mask.view(-1)
                index += size

        return masks              

    
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
