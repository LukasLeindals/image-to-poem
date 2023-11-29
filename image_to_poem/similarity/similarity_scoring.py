import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from bert_classifier import Bert_classifier
import numpy as np
import os

from image_to_poem.utils import load_json_file
from image_to_poem.similarity.similarity_data import get_dataloaders

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

# VARIABLES 
MAX_LENGTH = 200
NO_HIDDEN = 1
HIDDEN_DIM = 100

BATCH_SIZE = 5 
NUM_EPOCHS = 1
VAL_EPOCH = 5
LEARNING_RATE = 0.001

class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=1, hidden_dim: int = 100, max_length: int = 250):
        
        # models 
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = Bert_classifier(self.bert.config.hidden_size, no_hidden_layers, hidden_dim)
        
        # tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # other parameters 
        self.max_len = max_length 
        self.trained = False 
    
    def encode_input(self, caption: str, poem: str):
        return self.tokenizer.encode_plus(caption, poem, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
    
    def similarity(self, caption: str, poem: str) -> float:
        if not self.trained:
            print("The BERT Classifier has not been trained yet. Similarity might not be good.")
        
        # encode/tokenize caption and poem      
        encoding = self.encode_input(caption, poem)
        # send through BERT  
        bert_output = self.bert(**encoding)
        _, x = bert_output[0], bert_output[1]
        # send bert output through classifier 
        res = self.model(x)

        return res.item()
    
    def train_bert_classifier(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, 
                              val_epoch: int, learning_rate: float, save_path: str=None, verbose: bool=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer = Adam(self.model.parameters(), lr=learning_rate) 
        criterion = nn.BCELoss()
        
        # save training and validation loss
        losses = []
        val_losses = []
            
        for epoch_i in range(num_epochs):
            for (inp, lbl) in train_loader:
                # send input through BERT 
                bert_output = self.bert(**inp)
                _, cls_res = bert_output[0], bert_output[1]
                # fix label 
                target = torch.reshape(lbl,(-1,1)).float()
                # clear gradients 
                optimizer.zero_grad() 
                # forward pass 
                outputs = self.model(cls_res)
                # calc loss 
                loss = criterion(outputs, target)
                # update
                losses.append(loss)
                # get gradients 
                loss.backward()
                # update model params 
                optimizer.step()
            
            # validation epoch     
            if (epoch_i+1)%val_epoch == 0:
                
                for (val_inp, val_lbl) in val_loader:
                    # prepare input and label
                    val_bert_output = self.bert(**val_inp)
                    _, val_cls_res = val_bert_output[0], val_bert_output[1]
                    val_target = torch.reshape(val_lbl,(-1,1)).float()
                    
                    with torch.no_grad():
                        val_outputs = self.model(val_cls_res)
                        val_loss = criterion(val_outputs, val_target)
                        val_losses.append(val_loss)
                        
                if verbose:
                    print(f"Iteration: {epoch_i+1} \t Loss: {val_losses[-1]} \t") 
        
        # save model 
        if save_path is not None:
            self.model.save_model(save_path)
        
        if verbose:
            plt.figure()
            plt.plot(losses, "-", label="Training Loss")
            plt.plot([(i+1)*val_epoch for i in range(val_losses)], val_losses, ".-", label="Validation Loss")
            plt.xlabel("Batch Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
            
        self.trained = True 
        return losses, val_losses

def do_training(modelfolder):
    sim_model = BertSimilarityModel(no_hidden_layers=NO_HIDDEN, hidden_dim=HIDDEN_DIM, max_length=MAX_LENGTH)
    
    data = load_json_file("data/caption_poem.json")
    train_loader, val_loader = get_dataloaders(data[:-100], sim_model.encode_input, BATCH_SIZE, split=0.9)
    
    # train 
    modelfile = os.path.join(modelfolder, f"sim_model_{NO_HIDDEN}_{HIDDEN_DIM}.pt")
    print("~ starting training ~")
    loss, val_loss = sim_model.train_bert_classifier(train_loader, val_loader, NUM_EPOCHS, VAL_EPOCH, LEARNING_RATE, modelfile, verbose=True)
    
    np.savetxt(os.path.join(modelfolder,"loss.txt"),loss)
    np.savetxt(os.path.join(modelfolder,"val_loss.txt"),val_loss)

if __name__ == "__main__":
    do_training(f"models/similarity")