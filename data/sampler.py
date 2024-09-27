from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler


class FederatedSampler(Sampler):
    def __init__(
        self,
        dataset: Sequence,
        non_iid: int,
        n_clients: Optional[int] = 100,
       
    ):
        """Sampler for federated learning in both iid and non-iid settings.

        Args:
            dataset (Sequence): Dataset to sample from.
            non_iid (int): 0: IID, 1: Non-IID
            n_clients (Optional[int], optional): Number of clients. Defaults to 100.
            n_shards (Optional[int], optional): Number of shards. Defaults to 200.
        """
        self.dataset = dataset
        self.non_iid = non_iid
        self.n_clients = n_clients
      

        if self.non_iid:
            self.dict_users = self._sample_non_iid()
        else:
            self.dict_users = self._sample_iid()

    def _sample_iid(self) -> Dict[int, List[int]]:
        num_items = int(len(self.dataset) / self.n_clients)
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users

    def _sample_non_iid(self) -> Dict[int, List[int]]:
        print("sampling non_iid")
        min_size = 0
        K = 10
        N = len(self.dataset.targets)
        dict_users = {}
        
        while(min_size < 10):
            idx_batch = [[] for _ in range(self.n_clients) ]
            
            for k in range(K):
                #print(np.asarray(self.dataset.targets) == k)
                idx_k = np.where(np.asarray(self.dataset.targets) == k)[0]

                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(0.5, self.n_clients))
                proportions = np.array([p*(len(idx_j)<N/self.n_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            #print("min_size:", min_size)
        for j in range(self.n_clients):
            # print(f"Client {j}: len={len(idx_batch[j])}")
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]
                


        return dict_users 

    def set_client(self, client_id: int):
        self.client_id = client_id

    def __iter__(self):
        # fetch dataset indexes based on current client
        client_idxs = list(self.dict_users[self.client_id])
        for item in client_idxs:
            yield int(item)
