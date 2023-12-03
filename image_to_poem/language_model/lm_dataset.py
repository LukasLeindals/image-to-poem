from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch

class LMDataset(Dataset):
  def __init__(self, texts, tokenizer, max_length=768, max_texts=None):
    """
    Initializes a torch Dataset for language modeling.
    
    Parameters
    ----------
    texts (list of str): 
        List of texts to use for training.
    tokenizer (GPT2Tokenizer):
        An instance of the GPT2Tokenizer class.
    max_length (int, optional):
        Maximum length of the input sequence. Defaults to 768.
    max_texts (int, optional):
        Maximum number of texts to use. 
        Setting None means all available texts will be used.
        Defaults to None.
    """
    self.tokenizer = tokenizer
    
    # initialize lists for input ids and attention masks
    self.input_ids = []
    self.attn_masks = []

    # encode all texts
    for text in texts if max_texts is None else texts[:min(max_texts, len(texts))]:
      encodings_dict = tokenizer(tokenizer.bos_token+text+tokenizer.eos_token,
                                 truncation=True,
                                 max_length=max_length,
                                 padding="max_length")
      
      # append input ids and attention masks to lists
      self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
      self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]

def get_datasets(texts, tokenizer, train_frac, verbose = False, **kwargs):
    """
    Creates training and validation datasets for language modeling.
    
    Parameters
    ----------
    texts (list of str): 
        List of texts to use for training.
    tokenizer (GPT2Tokenizer):
        An instance of the GPT2Tokenizer class.
    train_frac (float):
        Fraction of texts to use for training.
    verbose (bool, optional):
        Whether to print verbose output. Defaults to False.
    **kwargs:
        Additional keyword arguments to pass to the LMDataset class.
    
    Returns
    -------
    train_dataset (LMDataset):
        Training dataset.
    val_dataset (LMDataset):
        Validation dataset.
    """
    
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
    """
    Creates a dataloader for language modeling.
    
    Parameters
    ----------
    dataset (LMDataset):
        Dataset to use.
    train (bool):
        Whether to create a training dataloader. If true a random sampler will be used, otherwise a sequential sampler.
    batch_size (int):
        Batch size.
    verbose (bool, optional):
        Whether to print verbose output. Defaults to False.
    **kwargs:
        Additional keyword arguments to pass to the DataLoader class.
    
    Returns
    -------
    dataloader (DataLoader):
        Dataloader for the data set.
    """
    # create dataloader
    dataloader = DataLoader(dataset, 
                            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset),
                            batch_size = batch_size,
                            **kwargs)
    
    # print number of batches
    if verbose:
        print(f"Number of batches: {len(dataloader)}")
        
    return dataloader