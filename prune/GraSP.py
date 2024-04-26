# pruning method of GraSP, from paper "PICKING WINNING TICKETS BEFORE TRAINING BY PRESERVING GRADIENT FLOW"

import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def _forward_pre_hooks(masks):
    def pre_hook(module, input):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            m = masks[str(module)]
            module.weight.data.mul_(m)
        else:
            raise NotImplementedError('Unsupported ' + m)
    return pre_hook

def _grad_hook(mask):
    def g_hook(grad):
        return grad * mask
    return g_hook

def _unregister_forward_mask(model):
    for m in model.modules():
        m._forward_pre_hooks = OrderedDict()

def _unregister_backward_mask(model):
    for m in model.modules():
        m._backward_hooks = OrderedDict()

def register_forward_mask(model, masks=None):
    _unregister_forward_mask(model)
    layer_dict = dict()
    for layer in model.modules():
        layer_dict[str(layer)] = layer
    assert masks is not None, 'Masks should be generated first.'
    for layer_name in masks.keys():
        assert layer_name in layer_dict.keys(), 'mask key not in origin model!'
        layer_dict[layer_name].register_forward_pre_hook(_forward_pre_hooks(masks))

def register_backward_mask(model, masks=None):
    _unregister_backward_mask(model)
    layer_dict = dict()
    for layer in model.modules():
        layer_dict[str(layer)] = layer
    assert masks is not None, 'Masks should be generated first.'
    for layer_name in masks.keys():
        assert layer_name in layer_dict.keys(), 'mask key not in origin model!'
        assert masks[layer_name].shape == layer_dict[layer_name].weight.shape, 'Mask shape mismatches the weight shape!'
        layer_dict[layer_name].weight.register_hook(_grad_hook(masks[layer_name]))

def pruning_by_mask(model, masks_dict):
    layer_dict = dict()
    for layer in model.modules():
        layer_dict[str(layer)] = layer
    for layer_name, mask in masks_dict.items():
        if isinstance(layer_dict[layer_name], nn.Linear) or isinstance(layer_dict[layer_name], nn.Conv2d):
            layer_dict[layer_name].weight.data.mul_(mask)

def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=10, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    # total_parameters = count_total_parameters(net)
    # fc_parameters = count_fc_parameters(net)

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    for it in range(num_iters):
        # print("Pruning (1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs[:N//2])/T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        # print("Pruning (2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    # print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    acceptable_score = threshold[-1]
    # print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[str(m)] = ((g / norm_factor) <= acceptable_score).float()

    # print total number of non-zero
    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks
