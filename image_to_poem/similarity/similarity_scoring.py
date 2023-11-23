import torch 
from transformers import BertTokenizer, BertModel
from bert_classifier import Bert_classifier


class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=2, hidden_dim: int = 100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = Bert_classifier(self.bert, no_hidden_layers, hidden_dim)
        
        self.trained = False 
    
    def tokenize_sequences(self, seq1 : str, seq2 : str) -> list:
        # tokenize sequences 
        # add BERT special tokens 
        # do padding 
        pass
    
    def similarity(self, seq1: str, seq2: str) -> float:
        if not self.trained:
            print("The bert classifier has not been trained yet")
        
        input = self.tokenize_sequences(seq1, seq2)
        out = self.model.forward(input)

        return out
    
    def train_bert_classifier(self):
        self.trained = True 
        # train self.model
    