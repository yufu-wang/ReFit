import torch
import numpy as np
from loguru import logger
from torch.utils.data import default_collate
from torch.utils.data.sampler import Sampler

from .base_dataset import BaseDataset
from data_config import H36M_MV_INDEX 
 
class MVDataset(torch.utils.data.Dataset):

    def __init__(self, dataset='h36m-mv', **kwargs):

        self.dataset = BaseDataset(dataset, **kwargs) 
        self.mv_idx = torch.tensor(np.load(H36M_MV_INDEX))

        self.length = len(self.mv_idx)


    def __getitem__(self, index):
        mv = self.mv_idx[index]
        items = []
        for idx in mv:
            item = self.dataset[idx]
            items.append(item)

        batch = default_collate(items)

        return batch

    def __len__(self):
        return self.length


class MV_Sampler(Sampler):
    '''
    Provide index to match multiple views from a regular dataset
    '''
    def __init__(self, mv_idx, shuffle=False):
        self.mv_idx = mv_idx
        self.dataset_perm = mv_idx.reshape(-1).tolist()
        self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)
