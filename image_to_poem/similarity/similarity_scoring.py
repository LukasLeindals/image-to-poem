import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from image_to_poem.similarity.bert_classifier import Bert_classifier
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
NUM_EPOCHS = 1
VAL_EPOCH = 1
LEARNING_RATE = 0.001
SAVE_EVERY = 1 # save model every x epochs

class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=1, hidden_dim: int = 100, max_length: int = 250):   
        """
        Initializes the BERT Similarity Model.

        Parameters
        ----------
        no_hidden_layers (int): 
            Number of hidden layers in the classifier 
        hidden_dim (int): 
            Number of nodes in the hidden layers of the classifier.
        max_length (int): 
            Maximum length of the encoded input to BERT. 
        """    
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
        """
        Encode a caption and poem for input to BERT as such: 
        [CLS] [CAP TOKEN 1] ... [CAP TOKEN N] [SEP] [POEM TOKEN 1] ... [POEM TOKEN M] [SEP] ([PAD] ...)

        Parameters
        ----------
        caption (str): 
            The caption. 
        poem (str): 
            The poem. 
        """  
        return self.tokenizer.encode_plus(caption, poem, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
    
    def similarity(self, caption: str, poem: str) -> float:
        """
        Calculate the similarity between a given caption and poem. 

        Parameters
        ----------
        caption (str): 
            The caption. 
        poem (str): 
            The poem. 
        """  
        if not self.trained:
            print("The BERT Classifier has not been trained yet. Similarity might not be good.")
        
        # encode/tokenize caption and poem      
        encoding = self.encode_input(caption, poem)
        
        # send to device
        encoding = {key: encoding[key].to(self.device) for key in encoding}
        
        # send through BERT  
        bert_output = self.bert(**encoding)
        _, x = bert_output[0], bert_output[1]
        
        # send bert output through classifier 
        res = self.model(x)

        return res.item()
    
    def train_bert_classifier(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = NUM_EPOCHS, 
                              val_epoch: int = VAL_EPOCH, learning_rate: float = LEARNING_RATE, save_path: str=None, verbose: bool=True):
        """
        A function to train the classifier. 

        Parameters
        ----------
        train_loader (DataLoader) : 
            Dataloader for the training data. 
        val_loader (DataLoader) : 
            Dataloader for the validation data. 
        num_epochs (int) :
            Number of epochs to train. 
        val_epoch (int) :
            How often to run a validation epoch. 
        learning_rate (float) : 
            Learning rate of the training for the optimizer. 
        save_path (str) :
            The path to where to save all the data from training. 
        verbose (bool) : 
            whether or not to print/save a training figure at the end of training. 
        """  
        save_folder = os.path.dirname(save_path) if save_path is not None else None
        # for training 
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
                "no_train_batch": len(train_loader),
                "no_val_batch" : len(val_loader),
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
                    print(f"Iteration: {epoch_i+1} \t Training Loss: {epoch_losses[-1]} \t Validation Loss: {val_losses[-1]} \t") 
        
        # save model 
        if save_path is not None:
            self.model.save_model(save_path)
            
        # send to cpu and convert to numpy array
        losses = [loss.cpu().detach().numpy() for loss in losses]
        val_losses = [val_loss.cpu().detach().numpy() for val_loss in val_losses]
        
        # plot loss
        if verbose:
            N = 25 
            smooth_losses = np.convolve(losses, np.ones(N)/N, mode='same')
            
            plt.figure()
            plt.plot(np.linspace(0,5,len(losses)), losses, "-", alpha=0.25, linewidth=0.5)
            plt.plot(np.linspace(0,5,len(losses)), smooth_losses, "-b", linewidth=0.5, label="Smoothed Training Loss (per batch step)")
            plt.plot(np.arange(5)+1, epoch_losses, ".-", linewidth=1.5, label="Training Loss")
            if len(val_losses) > 0:
                plt.plot(np.arange(5)+1,val_losses, ".-", linewidth=1.5, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.savefig(os.path.join(save_folder, "loss.png"))
            
        self.trained = True 
        return {
            "loss": losses,
            "val_loss": val_losses,
            "epoch_loss": epoch_losses,
        }
    
    @classmethod
    def from_model_dir(cls, model_dir: str, epoch: int = None):
        """
        A function to load a pre-trained saved model 

        Parameters
        ----------
        model_dir (str) : 
            Directory to a pretrained model 
        epoch (int) :
            Which epoch to load. If None, the last trained epoch is loaded. 
        """  
        # get params
        params = load_json_file(os.path.join(model_dir, "params.json"))
        
        # create model
        sim_model = cls(
            no_hidden_layers = params["no_hidden"], 
            hidden_dim = params["hidden_dim"],
            max_length = params["max_length"],
        )
        
        # load model
        model_name = f"sim_model_{NO_HIDDEN}_{HIDDEN_DIM}"
        if epoch is not None:
            model_name += f"_epoch_{epoch}"
        model_name += ".pt"
        sim_model.model.load_model(os.path.join(model_dir, model_name))

        # set model to trained
        sim_model.trained = True
        
        return sim_model

def do_training(model_name, modelfolder = "models/similarity", data_size = None, test_size = 100):
    """
    A function to set up and start a training of the similarity model. 

    Parameters
    ----------
    model_name (str) : 
        Model name. 
    modelfolder (str) :
        Folder where to save the training things. 
    data_size (int) : 
        If not none, this is the amount of data used for training and validation. 
    test_size (int) : 
        How much data to set aside for testing. 
    """  
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
    loss_dict = sim_model.train_bert_classifier(train_loader, val_loader, NUM_EPOCHS, VAL_EPOCH, LEARNING_RATE, modelfile, verbose=True)
    print(f"Finished training in {time.time()-t0} seconds")

    for key, val in loss_dict.items():
        np.savetxt(os.path.join(modelfolder,f"{key}.txt"), val)

if __name__ == "__main__":
    import datetime
    model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    do_training(model_name, data_size = 200)
    