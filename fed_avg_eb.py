from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os, time
import argparse
import shutil

from data import MNISTDataset, FederatedSampler
from models import CNN, MLP, vgg
from utils import arg_parser, average_weights, Logger
from EarlyBird import EarlyBird

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.schedule = [80,120]
        self.eb_epoch = 0
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.eb_model = None ##added
        # Loading training and testing dataset and is processed for clients to receive their own dataset
        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )
        
        self.is_best = False
        
        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
            
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=3, n_classes=10).to(self.device)
            self.target_acc = 0.99
            
        elif self.args.model_name == "vgg":
            self.root_model = vgg(dataset='cifar10', depth=19).to(self.device)
            self.target_acc = 0.85
            print("using vgg..")
            
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int

    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        train_set = MNISTDataset(root=root, train=True) ##TODO : it loads cifar10 despite using "MNISTDataset"
        test_set = MNISTDataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
        )

        train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        return train_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        model = copy.deepcopy(root_model)
        model.train()
        
        ###weight_decay is the new addition since early bird uses it. 
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = 1e-4
        )


        for epoch in range(self.args.n_client_epochs):
            """
            this is from Early Bird implementation where learning rate decreases at epoch = 80 and 120
            """
            if epoch in self.schedule: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1            
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                logits = model(data)
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples ##TODO

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )

        return model, optimizer.state_dict(), epoch_loss / self.args.n_client_epochs
        
    def save_checkpoint(self, state, is_best, epoch, filepath, eb_flag=False):
    
      """Save a model checkpoint.
      
      1) when new best model found , it prints "Saving the new best model at epoch {epoch+1} "
      2) when early bird found, it prints "Saving Early Bird Checkpoint for epoch {epoch+1}"
      3) when new best model as well as early bird , it prints "Saving the new best model at epoch {epoch+1}"
      
      """
      # Create the directory if it does not exist
      if not os.path.exists(filepath):
          os.makedirs(filepath)
  
      # Check if Early Bird checkpoint
      if eb_flag:
          filename = f'EB-30-{epoch+1}.pth.tar'
          full_file_path = os.path.join(filepath, filename)
          torch.save(state, full_file_path)
          print(f"\nSaving Early Bird checkpoint for epoch {epoch+1}...!")
  
      # If this is the best model, save it as 'model_best.pth.tar'
      elif is_best:
          filename = f'model_best.pth.tar'
          full_file_path = os.path.join(filepath, filename)
          torch.save(state, full_file_path)
          print(f"\nSaving the new best model at epoch {epoch+1}...!")

    def train(self, pre_eb = True) -> None:
        """Train a server model."""
  
        train_losses = []
        best_prec1 = 0.0

        found_eb = False
        early_bird_30 = EarlyBird(0.3)
        flag_30 = True

        for epoch in range(self.args.n_epochs):
            
            self.root_model.train() ## training session mode
            
            
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            ##after 4 epochs, this 'if' clause gets executed and runs only once.
            if pre_eb and early_bird_30.early_bird_emerge(self.root_model) and flag_30: 
                print("[early_bird_30] Found EB at epoch: "+str(epoch+1))
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.root_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': opt_state
                }, self.is_best, epoch, filepath=self.args.save, eb_flag = True)
                flag_30 = False
                self.eb_epoch = epoch + 1
                found_eb = True
                
            if(found_eb == True): #stop training once eb found so we can prune
              return
            
            
            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, opt_state, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
            
            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)
            
            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)
            
            total_loss, total_acc = self.test() # testing server model each epoch
            self.is_best = total_acc > best_prec1 ##checking for the best accuracy
            best_prec1 = max(total_acc, best_prec1)
            
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.root_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': opt_state
            }, self.is_best, epoch, filepath=self.args.save)


            if (epoch + 1) % self.args.log_every == 0: ##by default its = 1 
                # Test server model
    
                avg_train_loss = sum(train_losses) / len(train_losses)
 
                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )
                print("Best accuracy: "+str(best_prec1))
                
                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break

    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
    def count_zero_weights(self, s = ""):
        print(s)
        zero_channel_count = 0

        for layer in self.root_model.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):

                zero_weights = layer.weight == 0
                zero_channel_count += torch.sum(zero_weights).item()
        
        print("Number of zero channels: ", zero_channel_count)

    def actual_prune(self):
        #eb_model = './logs/EB-30-10.pth.tar'
        eb_model = f'./logs/EB-30-{self.eb_epoch}.pth.tar'
        
        if os.path.isfile(eb_model):
            print("=> loading checkpoint '{}'".format(eb_model))
            checkpoint = torch.load(eb_model)
            self.root_model.load_state_dict(checkpoint['state_dict'])

        else:
            print("=> no checkpoint found at '{}'".format(eb_model))
            exit()
        
        total = 0 ##to store the total number of weights in all the bn layers
        for m in self.root_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        # get the threshold index for each channel
        bn = torch.zeros(total)
        index = 0 
        for m in self.root_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size        
        y, _ = torch.sort(bn)
        thre_index = int(total * 0.3) # 0.3 since pruning weight is 0.3
        thre = y[thre_index]
        
        cfg = [] ##number of remaining channels
        cfg_mask = [] ##mask for each layer
        
        pruned = 0
        
        # PART I: pruning pre-processing starts
        for k, m in enumerate(self.root_model.modules()):
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                weight_copy = m.weight.data.abs().clone() # copy the absolute values of the weight
                mask = weight_copy.gt(thre.cuda()).float().cuda() # entering 1 (if above thre) else 0
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
        self.count_zero_weights(s = "Before removing zeroed out channels") 
        
        # PART II: actual pruning where zeroed weights (channels) are excluded. 
        
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]
        
        newmodel = vgg(dataset='cifar10', cfg = cfg)

        for [m0, m1] in zip(self.root_model.modules(), newmodel.modules()): 
            
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

        self.root_model = copy.deepcopy(newmodel)
        
        self.count_zero_weights(s="After removing zeroed out channels")
        
        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(self.args.save, 'pruned.pth.tar'))


        
        

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train() 
    print("EB was drawn...pruning starts now")
    fed_avg.actual_prune()
    print("Done pruning... training again")
    fed_avg.train(pre_eb=False)