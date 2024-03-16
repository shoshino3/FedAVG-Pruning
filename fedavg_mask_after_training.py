from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os

from data import MNISTDataset, FederatedSampler
from models import CNN, MLP, vgg
from utils import arg_parser, average_weights, Logger, calculate_density, density_weights, density_masks



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from snip import SNIP

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

LOG_INTERVAL = 20
INIT_LR = 0.1
WEIGHT_DECAY_RATE = 0.0005
EPOCHS = 250
REPEAT_WITH_DIFFERENT_SEED = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))


VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
}

class VGG_SNIP(nn.Module):
# """
# This is a base class to generate three VGG variants used in SNIP paper:
#     1. VGG-C (16 layers)
#     2. VGG-D (16 layers)
#     3. VGG-like

# Some of the differences:
#     * Reduced size of FC layers to 512
#     * Adjusted flattening to match CIFAR-10 shapes
#     * Replaced dropout layers with BatchNorm
# """

    def __init__(self, config, num_classes=10):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def make_layers(config, batch_norm=False):  # TODO: BN yes or no?
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_mnist_dataloaders(train_batch_size, val_batch_size):

    data_transform = Compose([transforms.ToTensor()])

    # Normalise? transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = MNIST("_dataset", True, data_transform, download=True)
    test_dataset = MNIST("_dataset", False, data_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        val_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    return train_loader, test_loader


def get_cifar10_dataloaders(train_batch_size, test_batch_size):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = CIFAR10('_dataset', False, test_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    return train_loader, test_loader


def mnist_experiment():

    BATCH_SIZE = 100
    LR_DECAY_INTERVAL = 25000

    # net = LeNet_300_100()
    # net = LeNet_5()
    net = LeNet_5_Caffe().to(device)

    optimiser = optim.SGD(
        net.parameters(),
        lr=INIT_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimiser, 30000, gamma=0.1)

    train_loader, val_loader = get_mnist_dataloaders(BATCH_SIZE, BATCH_SIZE)

    return net, optimiser, lr_scheduler, train_loader, val_loader


def cifar10_experiment():

    BATCH_SIZE = 128
    LR_DECAY_INTERVAL = 10000

    net = VGG_SNIP('D').to(device)

    optimiser = optim.SGD(
        net.parameters(),
        lr=INIT_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimiser, LR_DECAY_INTERVAL, gamma=0.05)

    train_loader, val_loader = get_cifar10_dataloaders(BATCH_SIZE,
                                                    BATCH_SIZE)  # TODO

    return net, optimiser, lr_scheduler, train_loader, val_loader
    

import os

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.max_accuracy = 0
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = Logger(args)
        # self.results_file = os.path.join(os.getcwd(), "results_file.txt")

        round = self.args.n_client_epochs

        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )

        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=3, n_classes=10).to(self.device)
            self.target_acc = 0.99
        elif self.args.model_name == "vgg":
            self.target_acc = 0.97
            self.root_model = vgg(dataset='cifar10', depth=19).to(self.device)
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int

        # Open the results file for writing
        # self.results_file = open(self.results_file, 'w')



    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.a
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        train_set = MNISTDataset(root=root, train=True)
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
        density_00 = density_weights(model)
        print("-------------------")
        print(f"density of root model: \n {[float(x) for x in density_00.values()]}")
        with open("res_with_sparsity.txt", 'a') as results_file:
            results_file.write(f"density of root model: \n {[float(x) for x in density_00.values()]}\n")

        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )
        # model = net
        density_0 = density_weights(model)
        print("-------------------")
        print(f"density of client model before PRUNING: \n {[float(x) for x in density_0.values()]}")
        with open("res_with_sparsity.txt", 'a') as results_file:
            results_file.write(f"density of client model before PRUNING: \n {[float(x) for x in density_0.values()]}\n")

        # # Pre-training pruning using SKIP
        keep_masks = SNIP(model, 0.5, train_loader, device)  # !!! check whether the model is the same as the one in the main function, or whether it is the same for all the clients. if so, only initialize once.
        apply_prune_mask(model, keep_masks) # move this to the first client which mean each round only do once for all the clients

        density_1 = density_weights(model)
        print("-------------------")
        print(f"density of client model after FIRST PRUNING: \n {[float(x) for x in density_1.values()]}")
        with open("res_with_sparsity.txt", 'a') as results_file:
            results_file.write(f"density of client model after FIRST PRUNING: \n {[float(x) for x in density_1.values()]}\n")

        for epoch in range(self.args.n_client_epochs): # epochs <= 30
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
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )
        density_2 = density_weights(model)
        print("-------------------")
        print(f"density of client model after TRAINING: \n {[float(x) for x in density_2.values()]}")
        with open("res_with_sparsity.txt", 'a') as results_file:
            results_file.write(f"density of client model after TRAINING: \n {[float(x) for x in density_2.values()]}\n")

        # keep_masks = SNIP(model, 0.5, train_loader, device)  # generate a new pruning mask
        apply_prune_mask(model, keep_masks)
        density_3 = density_weights(model)
        print("-------------------")
        print(f"density of client model after SECOND PRUNING: \n {[float(x) for x in density_3.values()]}")
        with open("res_with_sparsity.txt", 'a') as results_file:
            results_file.write(f"density of client model after SECOND PRUNING: \n {[float(x) for x in density_3.values()]}\n")

        return model, epoch_loss / self.args.n_client_epochs

    def train(self) -> None:
        """Train a server model."""
        train_losses = []

        for epoch in range(self.args.n_epochs): # epochs := rounds
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            density_r0 = density_weights(self.root_model)
            print("---------11----------")
            print(f"density of root model before training: \n {[float(x) for x in density_r0.values()]}")
            with open("res_with_sparsity.txt", 'a') as results_file:
                results_file.write(f"density of root model before training: \n {[float(x) for x in density_r0.values()]}\n")
            # Train clients
            self.root_model.train()
            density_r1 = density_weights(self.root_model)
            print("---------22----------")
            print(f"density of root model after training: \n {[float(x) for x in density_r1.values()]}")
            with open("res_with_sparsity.txt", 'a') as results_file:
                results_file.write(f"density of root model after training: \n {[float(x) for x in density_r1.values()]}\n")

            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                density_client = density_weights(client_model)
                print("---------33----------")
                print(f"density of client model after training: \n {[float(x) for x in density_client.values()]}")
                with open("res_with_sparsity.txt", 'a') as results_file:
                    results_file.write(f"density of client model after training: \n {[float(x) for x in density_client.values()]}\n")
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            # print("Updated weights: ", updated_weights)
            print("----------")

            # if not updated_weights:
            #     print("The state_dict is empty.")
            # else:
            #     print("The state_dict is not empty.")

            self.root_model.load_state_dict(updated_weights)

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print("-------------------")
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                self.logger.log(logs)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                if total_acc >= self.max_accuracy:
                    print(
                        f"-----> current max accuracy {self.target_acc} reached at round {epoch}! <-----"
                    )
                    self.max_accuracy = total_acc
                self.max_accuracy = max(self.max_accuracy, total_acc)
                print("max_acc,",self.max_accuracy)
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )
                
                # Open the results file in append mode and write the results
                with open("res_with_sparsity.txt", 'a') as results_file:
                    results_file.write(f"\n\nResults after {epoch + 1} rounds of training:\n")
                    results_file.write(f"---> Avg Training Loss: {avg_train_loss}\n")
                    results_file.write(f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n")
                    print("---------max acc----------")
                    self.max_accuracy = max(self.max_accuracy, total_acc)
                    results_file.write("max_acc, " + str(self.max_accuracy) + "\n")

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

if __name__ == "__main__":
    args = arg_parser()
    random_number = np.random.randint(1, 100)
    print("Random number: ", random_number)

    fed_avg = FedAvg(args)
    fed_avg.train()

    print("Number of client epochs: ",round)
