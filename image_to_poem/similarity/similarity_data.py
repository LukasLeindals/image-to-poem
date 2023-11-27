import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import torch

DATAPATH = "../../data/multim_poem.json"
PROCESSED_DATAPATH = "../../data/caption_poem.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: Issue! Update the Dataset to work with batch size > 1 
class CaptionPoemDataset(Dataset):
    def __init__(self, datadict : dict):
        self.data = datadict
        
        self.N_halfs = len(self.data) // 2 
        self.shuffle_idx = np.random.choice(a=self.N_halfs, size=self.N_halfs, replace=False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx < self.N_halfs:
            # match caption idx, with poem shuffle idx 
            match_idx = self.shuffle_idx[idx]
        else:
            match_idx = idx
        
        cap_poem = [self.data[idx]['caption'], self.data[match_idx]['poem']]
        label = 1 if idx == match_idx else 0 
        return cap_poem, label

def get_dataloaders(datadict, batch_size, split):
    # define dataset class 
    dataset = CaptionPoemDataset(datadict)
    
    # split data 
    train_size = int(split*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Number of samples for training =", train_size)
    print("Number of samples for validation =", val_size)
    
    # get dataloaders 
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    return train_dataloader, val_dataloader
