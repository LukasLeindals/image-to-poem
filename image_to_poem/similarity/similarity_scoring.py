import torch 
from transformers import BertTokenizer, BertModel
from bert_classifier import Bert_classifier


class BertSimilarityModel:
    def __init__(self, no_hidden_layers: int=2, hidden_dim: int = 100, max_length: int = 150):
        
        # models 
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = Bert_classifier(self.bert, no_hidden_layers, hidden_dim)
        
        # tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # other parameters 
        self.max_len = max_length 
        self.trained = False 
    
    def similarity(self, caption: str, poem: str) -> float:
        if not self.trained:
            print("The BERT classifier has not been trained yet. Similarity might not be good.")
            
        encoding = self.tokenizer.encode_plus(caption, poem, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        out = self.model.forward(encoding)

        return out
    
    def train_bert_classifier(self):
        self.trained = True 
        # train self.model



