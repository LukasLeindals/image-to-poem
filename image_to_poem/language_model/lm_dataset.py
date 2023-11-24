from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch

class LMDataset(Dataset):
  def __init__(self, texts, tokenizer, max_length=768, max_texts=None):
    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for text in texts if max_texts is None else texts[:min(max_texts, len(texts))]:

      encodings_dict = tokenizer(tokenizer.bos_token+text+tokenizer.eos_token,
                                 truncation=True,
                                 max_length=max_length,
                                 padding="max_length")
      
      self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
      self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]

def get_datasets(texts, tokenizer, train_frac, verbose = False, **kwargs):
    # create dataset
    dataset = LMDataset(texts, tokenizer, **kwargs)
    
    # split dataset
    train_size = int(train_frac * len(dataset))
    val_size = len(dataset) - train_size
    if verbose:
        print(f"Training set size: {train_size}")
        print(f"Validation set size: {val_size}")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def create_dataloader(dataset, train, batch_size, verbose = False, **kwargs):
    dataloader = DataLoader(dataset, 
                            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset),
                            batch_size = batch_size)
    if verbose:
        print(f"Number of batches: {len(dataloader)}")
        
    return dataloader