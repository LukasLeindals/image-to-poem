import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

class CaptionPoemDataset(Dataset):
    def __init__(self, datadict : list, encoder_fun):
        """
        Initializes the Caption-Poem Dataset.

        Parameters
        ----------
        datadict (list[dict]): 
            The raw caption-poem data. 
        encoder_fun (function): 
            An encoder function to encode a caption and poem. 
        """
        self.data = datadict
        self.encoder = encoder_fun 
                
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
        # collect caption and poem 
        cap = self.data[idx]['caption']
        poem = self.data[match_idx]['poem']
        # tokenize 
        X = self.encoder(cap, poem)
        # reshape input to correct size 
        X["input_ids"] = torch.flatten(X["input_ids"])
        X["token_type_ids"] = torch.flatten(X["token_type_ids"])
        X["attention_mask"] = torch.flatten(X["attention_mask"])
        # get label 
        y = 1 if idx == match_idx else 0 
        return X, y

def get_dataloaders(datadict : list, encoder_fun, batch_size : int, split : float):
    """
    Obtain a training and validation dataloader for training the BERT similarity model 

    Parameters
    ----------
    datadict (list[dict]): 
        The raw caption-poem data. 
    encoder_fun (function): 
        An encoder function to encode a caption and poem. 
    batch_size (int):
        The batch size of for both dataloaders.
    split (float):
        How to split the data: split% of the data to use for training and (1-split)% for validation. 
    """
    # define dataset class 
    dataset = CaptionPoemDataset(datadict, encoder_fun)
    
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
