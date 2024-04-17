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
from collections import defaultdict
from data import MNISTDataset, FederatedSampler
from models import CNN, MLP, vgg
from utils import arg_parser, average_weights, Logger
from EarlyBird_modified import EarlyBird

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        
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
        
        self.client_masks = {}
        
        
        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
            
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=3, n_classes=10).to(self.device)
            self.target_acc = 0.85
            
        elif self.args.model_name == "vgg":
            self.root_model = vgg(dataset='cifar10', depth=19).to(self.device)
            self.target_acc = 0.82
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
        self, eb_client, pre_eb, model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """

        model = copy.deepcopy(model)  
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = 1e-4
        )

            
        for epoch in range(self.args.n_client_epochs):
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            loss = 0.0
            
   
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
            epoch_acc = epoch_correct / epoch_samples 
            

            if pre_eb and eb_client.early_bird_emerge(model): #eb_client = class instance
                
                print("[early_bird_30] Found EB at epoch: {} for client_id = {} ".format(epoch+1, client_idx))
                """ 
                Once EB found for a client, do this - 

                1) get the mask for that client and prune it using self.fake_prune(), 
                2) register the mask to the hook
                3) keep training the pruned client model till the n_client_epochs is reached 
                4) then return this trained pruned local model inside train() to append it to clients_models[] 
                to store this client's stateto update the server          
                """

                model, client_mask = self.fake_prune(model) #and then fake prune using the mask
                eb_client.register_mask(model, client_mask) #register mask
                pre_eb = False

                 
            """
            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )
            """
         # Do another round of pruning so that it will eliminate the non-zero introduced by back propagation 
         #but this time using the mask that was generated instead of generating a new one. There for mask == client_mask
         # instead of "mask == None"    
        
          
        model, client_mask = self.fake_prune(model, post_prune = True)
        
        
        return model, epoch_loss / self.args.n_client_epochs
        


    def train(self) -> None:
        """Train a server model."""
        train_losses = []
        best_prec1 = 0.0
     
        epoch_acc_tracker = np.zeros((self.args.n_epochs, 2))
        
        
        for epoch in range(self.args.n_epochs):
            
            self.root_model.train() ## training session mode
            
            clients_models = []
            clients_losses = []
            
            if((epoch + 1) % 100 == 0):
                self.args.lr = 0.95*self.args.lr
                print("Value of lr is {} at epoch {}".format(self.args.lr, epoch))            


            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

        
            for client_idx in idx_clients:
                print(" using client_id = {} ".format(client_idx))
                eb_client = EarlyBird(0.3)      
                
                self.train_loader.sampler.set_client(client_idx) 
                    
                client_model, client_loss = self._train_client(
                    eb_client = eb_client,
                    pre_eb = True,
                    model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx

                )

                #by the time the client model is appended, it has been pruned
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
                
            print("training server now")    
                
            # Update server model based on clients models
            updated_weights = average_weights(clients_models) #averages sparse weights from clients
            self.root_model.load_state_dict(updated_weights) 
            
            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)
            
            total_loss, total_acc = self.test() # testing server model each epoch
            self.is_best = total_acc > best_prec1 ##checking for the best accuracy
            best_prec1 = max(total_acc, best_prec1)


            if (epoch + 1) % self.args.log_every == 0: ##by default its = 1 
                # Test server model
    
                avg_train_loss = sum(train_losses) / len(train_losses)
                
                if total_acc >= self.target_acc and self.reached_target_at is None:
                   
                    print("target accuracy = {} reached at = {}".format(total_acc, epoch+1))
                    return
      

                # to visualize learning curve
                epoch_acc_tracker[epoch,0] = epoch + 1 
                epoch_acc_tracker[epoch,1] = total_acc
                
                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )
                print("Best accuracy: "+str(best_prec1))
  

        np.savetxt(os.path.join(self.args.save, 'record_client_30.txt'), epoch_acc_tracker, fmt = '%10.5f', delimiter=',')
    
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
        
    def count_zero_weights(self, model, s = ""):
            print(s)
            zero_channel_count = 0
    
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
    
                    zero_weights = layer.weight == 0
                    zero_channel_count += torch.sum(zero_weights).item()
            
            print("Number of zero channels: ", zero_channel_count)
    
    def fake_prune(self, model, post_prune = False):
    
        if post_prune:
          print("post prunning training for local clients")
        #step 1: calculate the local client mask first 
        
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
        thre_index = int(total * 0.3)
        thre = y[thre_index]
        
        cfg = [] ##number of remaining channels
        cfg_mask = [] ##mask for each layer
        
        pruned = 0
        
        #step 2: fake pruning starts
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                weight_copy = m.weight.data.abs().clone() # copy the absolute values of the weight
                mask = weight_copy.gt(thre.cuda()).float().cuda() # entering 1 (if above thre) else 0
                pruned = pruned + mask.shape[0] - torch.sum(mask) # no. of pruned channels
                print(m.weight.data.shape)
                print(mask.shape)
                m.weight.data.mul_(mask) # put mask on the weights
                m.bias.data.mul_(mask) # put mask on the bias
  
                
        #count the number of zero channels: 

        self.count_zero_weights(model, s = "Count zero channels") 
        return model, mask


if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train() 
    