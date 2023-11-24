from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import random
import time
import tqdm
import datetime

from image_to_poem.language_model.lm_dataset import get_datasets, create_dataloader
from image_to_poem.utils import format_time

class GPT2Model:
    def __init__(self, pretrained_model = "gpt2", name = None, device: torch.device = None, verbose = False, seed = 42):
        self.pretrained_model = pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.set_seed(seed)
        self.verbose = verbose
        
        # set name
        self.name = name if name is not None else "model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Loading GPT2 Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model, 
                                          bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>',)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.load_model()
        
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def load_model(self):
        # Load model configuration
        config = GPT2Config.from_pretrained(self.pretrained_model)

        # Create model instance and set embedding length
        model = GPT2LMHeadModel.from_pretrained(
            self.pretrained_model, 
            config=config)
        
        
        model.resize_token_embeddings(len(self.tokenizer))

        # Running the model on GPU
        self.model = model.to(self.device)
        
    def generate(self, prompt: str = None, max_length: int = 100, num_return_sequences: int = 1):
        prompt = self.tokenizer.bos_token if prompt is None else prompt
        
        # set eval mode
        self.model.eval()
        
        # encode context the generation is conditioned on       
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        sample_outputs = self.model.generate(
                                        **encoded_prompt,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        do_sample=True, 
                                        max_length=max_length, 
                                        top_k=50, 
                                        top_p=0.95, 
                                        num_return_sequences=num_return_sequences)
        
        # decode output
        input_len = encoded_prompt["input_ids"].size()[-1] # length of input prompt - used to exclude input from output
        outputs = [self.tokenizer.decode(output[input_len:], skip_special_tokens=True) for output in sample_outputs]
        
        return outputs
     
    
if __name__ == "__main__":   
    # setup
    # pretrained_model = "gpt2"
    pretrained_model = "models/language_models/model_20231123_190606/model/"
    
    # create model
    gpt2_model = GPT2Model(verbose = True, pretrained_model = pretrained_model)   
    
    print(gpt2_model.generate("what is natural language processing?"))