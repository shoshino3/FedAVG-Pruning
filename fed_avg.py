from typing import Any, Dict, List, Optional, Tuple
import copy
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MNISTDataset, CIFAR10Dataset, FederatedSampler
from models import CNN, MLP, vgg
from utils import Logger, density_weights, density_masks, plot_data
from prune.GraSP import GraSP, register_mask, _unregister_mask, pruning_by_mask


class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.scheduler = [int(x) for x in self.args.lr_scheduler.split(",")] if self.args.lr_scheduler else []
        self.device = torch.device(
            f"cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
        )
        self.logger = Logger(args)
        self.accuracy, self.loss = [], []

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
            self.root_model = CNN(n_channels=3 if self.args.dataset=="cifar10" else 1, n_classes=10).to(self.device)
            self.target_acc = 0.99
        elif self.args.model_name == "vgg":
            assert self.args.dataset != "mnist", "do not support mnist dataset on vgg now."
            self.root_model = vgg(dataset=self.args.dataset, depth=19).to(self.device)
            self.target_acc = 0.85
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int
        self.client_masks = defaultdict(list)

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
        if self.args.dataset == "mnist":
            train_set = MNISTDataset(root=root, train=True)
            test_set = MNISTDataset(root=root, train=False)
        elif self.args.dataset == "cifar10":
            train_set = CIFAR10Dataset(root=root, train=True)
            test_set = CIFAR10Dataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
        )

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        # create a new dataloader to load a few samples for GraSP
        # TODO: maybe not work for Non-IID, need to check it
        self.GraSP_train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

        return train_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int, n_round: int, pruning_client:int,
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.
            n_round (int): number of epoch(round) of the server

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        client_acc, client_loss = [], []
        model = copy.deepcopy(root_model)
        # if self.args.pruning and self.args.pruning_side == "server":
        #     # deepcopy will also copy the hook function, to remove the hook function on the client model
        #     _unregister_mask(model)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum
        )     

        # for epoch #0, we use GraSP to find Sparse Mask, it need to use 10 x num_classes of data
        if client_idx not in self.client_masks.keys() and pruning_client:
            # how many iteration we achieve the final sparsity, now we single-shot
            num_iterations = 1
            pruning_ratio = args.pruning_ratio
            assert pruning_ratio >= 0.0 and pruning_ratio <1.0, "Pruning ratio should be in [0.0, 1.0)."
            if pruning_ratio > 0.0:
                ratio = 1 - (1 - pruning_ratio) ** (1.0 / num_iterations)
                self.client_masks[client_idx].append(GraSP(model, ratio, self.GraSP_train_loader, self.device,
                              num_classes = 10,
                              samples_per_class = 10,
                              num_iters = 1))
            # apply fake pruning: setting weights to zero
            register_mask(model, self.client_masks[client_idx][-1])
        
        # apply pruning after get the global model first
        # if pruning_client:
        #     pruning_by_mask(model, self.client_masks[client_idx][-1])

        for epoch in range(self.args.n_client_epochs):
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
            client_loss.append(epoch_loss)
            client_acc.append(epoch_acc)

        print(
            f"Client #{client_idx} | Avg Loss: {sum(client_loss)/len(client_loss):.4f} | Avg Acc: {sum(client_acc)/len(client_acc):.4f}",
            # end="\r",
        )

        # Do another round of pruning so that it will eliminate the non-zero introduced by back propagation
        if pruning_client:
            pruning_by_mask(model, self.client_masks[client_idx][-1])
            # density = density_weights(model)
            # print(f"density of client model after local training: \n {[float(x) for x in density.values()]}")

        return model, epoch_loss / self.args.n_client_epochs

    def train(self) -> None:
        """Train a server model."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            # Train clients
            self.root_model.train()

            # Set up global pruner and get mask
            # import ipdb; ipdb.set_trace()
            if args.pruning and args.pruning_side == "server":
                self.global_masks = GraSP(self.root_model, args.pruning_ratio, self.GraSP_train_loader, device=self.device)
                # register mask to the root model so that the test will have the exactly same mask
                register_mask(self.root_model, self.global_masks)
            
            # Learning rate scheduler
            if epoch in self.scheduler: 
                print(f"Changing learning rate: from {self.args.lr} to {self.args.lr*0.1}")
                self.args.lr = self.args.lr * 0.1

            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                    n_round=epoch,
                    pruning_client=True if (args.pruning and args.pruning_side=="client") else False,
                )
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # TODO: Pruning on the client side before collect all weights, since the mask is all same and this will save upload bandwidth
            # Need to implement pruning methods using client_model.state_dict() instead of client_model as in "pruning_by_mask()"


            # Update server model based on clients models
            updated_weights = self.average_weights(clients_models)
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
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                self.logger.log(logs)
                self.accuracy.append(total_acc)
                self.loss.append(total_loss)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss:.4f}")
                print(
                    f"---> Avg Test Loss: {total_loss:.4f} | Avg Test Accuracy: {total_acc:.4f}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break
        
        if self.args.draw_curve:
            plot_data(self.accuracy, "accuracy curve")
            plot_data(self.loss, "loss curve")

    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        # density = density_weights(self.root_model)
        # print(f"density of global model after aggregation: \n {[float(x) for x in density.values()]}")

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

    def average_weights(self, weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        weights_avg = copy.deepcopy(weights[0])

        for key in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[key] += weights[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(weights))

        return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("-m", "--model_name", type=str, default="vgg")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])

    parser.add_argument("-i", "--non_iid", type=int, default=0)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("-se", "--n_epochs", type=int, default=1000)
    parser.add_argument("-ce", "--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("-lrs", "--lr_scheduler", type=str, default="")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=str, default="gpu")

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    # customized arguments
    parser.add_argument("-p", "--pruning", action="store_true", help="whether to prune")
    parser.add_argument("-ps", "--pruning_side", type=str, default="client", choices=["client", "server"])
    parser.add_argument("-pr", "--pruning_ratio", type=float, default=0.9)
    parser.add_argument("-dc", "--draw_curve", action="store_true", help="whether to draw the accuracy curve")

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train()
