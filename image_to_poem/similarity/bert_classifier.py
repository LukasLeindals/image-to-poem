import torch
from torch import nn
import os

class Bert_classifier(nn.Module):
    def __init__(self, input_dim, no_hidden_layers: int=2, hidden_dim: int = 100):
        super(Bert_classifier, self).__init__()
        # self.bert = bert
        # do not train bert parameters!
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # define linear model on top of bert 
        layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU()]
        for _ in range(no_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.appen(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        
        self.final_layer = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        # send through bert 
        # out = self.bert(**bert_input)
        # _, x = out[0], out[1]
        
        # send through FFNN
        x = self.hidden_layers(x)
        x = self.final_layer(x)
        x = self.sigm(x)
        
        return x
    
    def save_model(self, modelpath):
        torch.save(self.state_dict(), modelpath)
    
    def load_model(self, modelpath):
        if not os.path.exists(modelpath):
            raise FileNotFoundError("Model path does not exist.")
        self.load_state_dict(torch.load(modelpath))
        
