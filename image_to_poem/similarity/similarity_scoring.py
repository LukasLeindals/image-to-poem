import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from bert_classifier import Bert_classifier
import numpy as np
import os
import tqdm
import time
from image_to_poem.utils import save_json_file
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
NUM_EPOCHS = 5
VAL_EPOCH = 1
LEARNING_RATE = 0.001
SAVE_EVERY = 1 # save model every x epochs



class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=1, hidden_dim: int = 100, max_length: int = 250):       
        # models 
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = Bert_classifier(self.bert.config.hidden_size, no_hidden_layers, hidden_dim)
        
        # tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # get device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set device
        self.to_device(self.device)
        
        # other parameters 
        self.max_len = max_length 
        self.trained = False 
    
    def to_device(self, device: torch.device):
        self.device = device 
        self.bert.to(self.device)
        self.model.to(self.device)
    
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
    
    def train_bert_classifier(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = NUM_EPOCHS, 
                              val_epoch: int = VAL_EPOCH, learning_rate: float = LEARNING_RATE, save_path: str=None, verbose: bool=True):
        save_folder = os.path.dirname(save_path) if save_path is not None else None
        optimizer = Adam(self.model.parameters(), lr=learning_rate) 
        criterion = nn.BCELoss()
        
        # save params
        if save_folder is not None:
            save_json_file(os.path.join(save_folder, "params.json"), {
                "max_length": MAX_LENGTH, 
                "no_hidden": NO_HIDDEN, 
                "hidden_dim": HIDDEN_DIM, 
                "batch_size": BATCH_SIZE, 
                "num_epochs": num_epochs, 
                "val_epoch": val_epoch, 
                "learning_rate": learning_rate,
            })
        
        # save training and validation loss
        losses = []
        epoch_losses = []
        val_losses = []
        
        self.model.train()
        for epoch_i in range(num_epochs):
            print(f"Training epoch {epoch_i+1} of {num_epochs}")
            epoch_loss = 0
            for (inp, lbl) in tqdm.tqdm(train_loader):                
                # send to device
                lbl = lbl.to(self.device)
                inp = {key: inp[key].to(self.device) for key in inp}
                
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
                epoch_loss += loss.item()
               
                # update
                losses.append(loss)
                
                # get gradients 
                loss.backward()
                
                # update model params 
                optimizer.step()
                
            # append mean epoch loss
            epoch_losses.append(epoch_loss/len(train_loader))
            
            # save checkpoint
            if save_path is not None and (epoch_i+1)%SAVE_EVERY == 0:
                path, ext = os.path.splitext(save_path)
                path = path + f"_epoch_{epoch_i+1}" + ext
                self.model.save_model(path)
            
            # validation epoch     
            if (epoch_i+1)%val_epoch == 0:
                self.model.eval()
                
                val_loss = 0
                for (val_inp, val_lbl) in val_loader:
                    # send to device
                    val_lbl = val_lbl.to(self.device)
                    val_inp = {key: val_inp[key].to(self.device) for key in val_inp}
                    
                    # prepare input and label
                    val_bert_output = self.bert(**val_inp)
                    _, val_cls_res = val_bert_output[0], val_bert_output[1]
                    val_target = torch.reshape(val_lbl,(-1,1)).float()
                    
                    with torch.no_grad():
                        val_outputs = self.model(val_cls_res)
                        val_loss += criterion(val_outputs, val_target)
                
                # append mean validation loss
                val_losses.append(val_loss/len(val_loader))
                
                # set model back to train mode
                self.model.train()
                        
                if verbose:
                    print(f"Iteration: {epoch_i+1} \t Loss: {val_losses[-1]} \t") 
        
        # save model 
        if save_path is not None:
            self.model.save_model(save_path)
            
        
        # send to cpu and convert to numpy array
        losses = [loss.cpu().detach().numpy() for loss in losses]
        val_losses = [val_loss.cpu().detach().numpy() for val_loss in val_losses]
        
        # plot loss
        if verbose:
            plt.figure()
            plt.plot(np.arange(1, len(train_loader)*num_epochs + 1), losses, "-", label="Training Loss (per batch step)")
            plt.plot(np.arange(1, len(epoch_losses)+1)*len(train_loader), epoch_losses, ".-", label="Training Loss (mean per epoch)")
            if len(val_losses) > 0:
                plt.plot(np.arange(1, len(val_losses)+1)*len(train_loader), val_losses, ".-", label="Validation Loss")
            plt.xlabel("Batch Step")
            plt.ylabel("Loss")
            plt.legend()

            plt.savefig(os.path.join(save_folder, "loss.png"))
            
        self.trained = True 
        return losses, val_losses

def do_training(model_name, modelfolder = "models/similarity", data_size = None, test_size = 100):
    modelfolder = os.path.join(modelfolder, model_name)
    os.makedirs(modelfolder, exist_ok=True)
    
    sim_model = BertSimilarityModel(no_hidden_layers=NO_HIDDEN, hidden_dim=HIDDEN_DIM, max_length=MAX_LENGTH)
    
    data = load_json_file("data/caption_poem.json")
    data = data[:-test_size] # skip last 100 poems for testing
    data = data[:min(data_size, len(data))] if data_size is not None else data
    train_loader, val_loader = get_dataloaders(data, sim_model.encode_input, BATCH_SIZE, split=0.9)
    
    # train 
    modelfile = os.path.join(modelfolder, f"sim_model_{NO_HIDDEN}_{HIDDEN_DIM}.pt")
    print(f"~ starting training on {sim_model.device}~")
    t0 = time.time()
    loss, val_loss = sim_model.train_bert_classifier(train_loader, val_loader, NUM_EPOCHS, VAL_EPOCH, LEARNING_RATE, modelfile, verbose=True)
    print(f"Finsihed training in {time.time()-t0:.2} seconds")
    
    np.savetxt(os.path.join(modelfolder,"loss.txt"),loss)
    np.savetxt(os.path.join(modelfolder,"val_loss.txt"),val_loss)

if __name__ == "__main__":
    import datetime
    
    model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    do_training(model_name, data_size = None)