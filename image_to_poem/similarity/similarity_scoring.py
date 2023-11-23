import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from bert_classifier import Bert_classifier
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()


class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=2, hidden_dim: int = 100, max_length: int = 150):
        
        # models 
        bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = Bert_classifier(bert, no_hidden_layers, hidden_dim)
        
        # tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # other parameters 
        self.max_len = max_length 
        self.trained = False 
    
    def tokenize(self, caption: str, poem: str):
        return self.tokenizer.encode_plus(caption, poem, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
    
    def similarity(self, caption: str, poem: str) -> float:
        if not self.trained:
            print("The BERT classifier has not been trained yet. Similarity might not be good.")
            
        encoding = self.tokenizer(caption, poem)
        out = self.model.forward(encoding)

        return out
    
    def train_bert_classifier(self, train_loader: DataLoader, val_loader: DataLoader, 
                              num_epochs, val_epoch: int, learning_rate: float, verbose: bool=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer = Adam(self.model.parameters(), lr=learning_rate) 
        criterion = nn.CrossEntropyLoss()
        
        # for 
        if verbose:
            losses = []
            val_losses = []
            
        for epoch_i in range(num_epochs):
            for (input, label) in train_loader:
                # load data input 
                caption, poem = input[0], input[1]
                # tokenize 
                bert_input = self.tokenize(caption, poem).to(device)
                # clear gradients 
                optimizer.zero_grad() 
                # forward pass 
                outputs = self.model(bert_input)
                # calc loss 
                loss = criterion(outputs, label)
                if verbose:
                    losses.append(loss.item())
                # get gradients 
                loss.backward()
                # update model params 
                optimizer.step()
                
                if (epoch_i+1)%val_epoch == 0:
                    
                    for (val_input, val_label) in val_loader:
                        # load data input 
                        val_caption, val_poem = val_input[0], val_input[1]
                        # tokenize 
                        bert_val_input = self.tokenize(val_caption, val_poem).to(device)
                        
                        with torch.no_grad():
                            val_outputs = self.model.forward(bert_val_input)
                            val_loss = criterion(val_outputs, val_label)
                            
                    if verbose:
                        val_losses.append(val_loss)
                        print(f"Iteration: {epoch_i+1} \t Loss: {val_loss.item()} \t") 
        
        if verbose:
            plt.figure()
            plt.plot(np.arange(num_epochs), losses, "-", label="Train loss")
            plt.plot([i*val_epoch for i in range(num_epochs//val_epoch)], val_losses, ".-", label="Val loss")
            plt.xlabel("Epoch number")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()                
        
        self.trained = True 

