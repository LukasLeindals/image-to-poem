from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import numpy as np
import random
import datetime
import os


class GPT2Model:
    def __init__(self, pretrained_model = "gpt2", name = None, device: torch.device = None, verbose = False, seed = 42):
        """
        Initializes the GPT2 language model.

        Parameters
        ----------
        pretrained_model (str): 
            Path to the pretrained GPT2 model or the name of the pretrained model.
        name (str, optional): 
            Name of the model. If not provided, it will be set to "model_" followed by the current date and time.
        device (torch.device, optional): 
            Device to use for running the model. If not provided, it will use CUDA if available, otherwise CPU.
        verbose (bool, optional): 
            Whether to print verbose output. Defaults to False.
        seed (int, optional): 
            Seed for random number generation. Defaults to 42.
        """
        self.pretrained_model = os.path.join(pretrained_model, "model/") if os.path.isdir(pretrained_model) else pretrained_model
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
        """
        Sets the seed for random number generation for reproducibility.
        
        Parameters
        ----------
        seed (int, optional): 
            Seed for random number generation. Defaults to 42.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def load_model(self):
        """
        Loads the a pretrained GPT2 model from the set directory.
        """
        # Load model configuration
        config = GPT2Config.from_pretrained(self.pretrained_model)

        # Create model instance and set embedding length
        model = GPT2LMHeadModel.from_pretrained(
            self.pretrained_model, 
            config=config)
        
        # set embedding length of model to match the tokenizer
        model.resize_token_embeddings(len(self.tokenizer))

        # Running the model on GPU
        self.model = model.to(self.device)
        
    def generate(self, prompt: str = None, max_length: int = 100, num_return_sequences: int = 1, **kwargs):
        """
        Generates a poem using the GPT2 model.
        
        Parameters
        ----------
        prompt (str): 
            Prompt to start the generation from. Defaults to None.
        max_length (int):
            Maximum length of the generated poem. Defaults to 100.
        num_return_sequences (int):
            Number of poems to generate. Defaults to 1.
        **kwargs:
            Additional keyword arguments passed to the model.generate() method. See [this blog](https://huggingface.co/blog/how-to-generate) for more information on available parameters.
        """
        
        # update default parameters
        kwargs["num_return_sequences"] = num_return_sequences 
        kwargs["max_length"] = max_length
        kwargs["do_sample"] = True if "do_sample" not in kwargs else kwargs["do_sample"]
        kwargs["top_k"] = 50 if "top_k" not in kwargs else kwargs["top_k"]
        kwargs["top_p"] = 0.95 if "top_p" not in kwargs else kwargs["top_p"] # use nucleus sampling -this means that the model will sample between all the tokens whose cumulative probability exceeds the probability p
        
        
        # set prompt
        prompt = self.tokenizer.bos_token if prompt is None else prompt
        
        # set eval mode
        self.model.eval()
        
        # encode context the generation is conditioned on       
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # sample the output
        sample_outputs = self.model.generate(**encoded_prompt, pad_token_id=self.tokenizer.pad_token_id, **kwargs)
        
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
    
    # use test prompt and generate poem
    print(gpt2_model.generate("what is natural language processing?"))