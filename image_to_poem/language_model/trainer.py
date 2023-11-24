import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import random
import json
import time
import datetime
import os
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup #,AdamW

import torch
torch.manual_seed(64)
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW


from image_to_poem.utils import format_time, update_param_dict
from image_to_poem.language_model.lm_dataset import get_datasets, create_dataloader
from image_to_poem.data.kaggle_poems import KagglePoems

def set_seed(seed_val = 42):
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

class Trainer:
	params = {
		"epochs" : 4,
		"sample_every" : 0.5,
		"save_every" : None,
		"batch_size" : 8,
		"optimizer" : {
			"lr" : 5e-4,
			"eps" : 1e-8,
		},
		"scheduler" : {
			"num_warmup_steps" : 1e2,
		},
		"sample" : {
			"do_sample": True,   
			"top_k": 50, 
			"max_length": 200,
			"top_p": 0.95, 
			"num_return_sequences":1,
		},
		"dataset" : {
			"train_frac": 0.9, 
   			"max_length": 100, 
      		"max_texts": None,
		},
	}
	
	def __init__(self, lm_model, data, train_params = {}, verbose = False):
		self.verbose = verbose 
  
		# save lm model and extract necesary components
		self.device = lm_model.device
		self.tokenizer = lm_model.tokenizer
		self.model = lm_model.model
  
		# create output dir
		self.create_output_dir(lm_model)
  
		# setup training
		self.setup_training(data, train_params)
  
		# some variables
		self.current_epoch = 0
		
  
	def create_output_dir(self, lm):
		self.output_dir = "models/language_models/" + lm.name + "/"
		os.makedirs(self.output_dir, exist_ok = True)
  
	def save(self):
		model_dir = self.output_dir + "model/"
		print(f"Saving model to '{model_dir}' ...")
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		model_to_save.save_pretrained(model_dir)
		self.tokenizer.save_pretrained(model_dir)
		
	def sample(self):
		# create folder for samples
		sample_dir = self.output_dir + "samples/"
		os.makedirs(sample_dir, exist_ok = True)
  
		# turn on eval mode
		self.model.eval()

		# create samples
		sample_outputs = self.model.generate(bos_token_id=random.randint(1,30000), 
                                       pad_token_id=self.tokenizer.pad_token_id,
                                       **self.params["sample"])
  
		# decode and save sample
		for i, sample_output in enumerate(sample_outputs):
			sample_ = self.tokenizer.decode(sample_output, skip_special_tokens=True)
			with open(sample_dir + f"epoch_{self.current_epoch}_batch_{self.current_batch_no}_{i}.txt", "w") as file:
				file.write(sample_)

		# go back to train mode
		self.model.train()
  
	def load_data(self, data):
		self.train_dataset, self.val_dataset = get_datasets(data, self.tokenizer, **self.params["dataset"], verbose=self.verbose)
		self.train_dataloader = create_dataloader(self.train_dataset, train=True, batch_size=self.params["batch_size"], verbose=self.verbose)
		self.val_dataloader = create_dataloader(self.val_dataset, train=False, batch_size=self.params["batch_size"], verbose=self.verbose)

	def validate(self):
		t0 = time.time()
		self.model.eval()
  
		total_eval_loss = 0
		nb_eval_steps = 0

		for batch in self.val_dataloader:
			b_input_ids = batch[0].to(self.device)
			b_labels = batch[0].to(self.device)
			b_masks = batch[1].to(self.device)
				
			with torch.no_grad():        

				outputs  = self.model(b_input_ids,  
								attention_mask = b_masks,
								labels=b_labels)
					
				loss = outputs[0]  
					
				batch_loss = loss.item()
				total_eval_loss += batch_loss   

		avg_val_loss = total_eval_loss / len(self.val_dataloader)  
		val_time = format_time(time.time() - t0)    
		print(f'Validation loss: {avg_val_loss}. Validation Time: {val_time}')
  
		return avg_val_loss, val_time

	def train_one_epoch(self):
		self.current_epoch += 1
		self.current_batch_no = 0
  
  
		t0 = time.time()
		total_train_loss = 0
		self.model.train()

		# Labels are shifted by 1 timestep
		pbar = tqdm(enumerate(self.train_dataloader), 
            	desc = f"Training epoch {self.current_epoch} out of {self.params['epochs']}", 
             	total = len(self.train_dataloader))
		for step, batch in pbar:
			self.current_batch_no += 1
			b_input_ids = batch[0].to(self.device)
			b_labels = batch[0].to(self.device)
			b_masks = batch[1].to(self.device)

			self.model.zero_grad()

			outputs = self.model(b_input_ids,
							labels=b_labels,
							attention_mask=b_masks)
			
			loss = outputs[0]

			batch_loss = loss.item()
			total_train_loss += batch_loss

			# Sampling every x steps
			sample_every = sample_every if self.params["sample_every"] > 1 else int(self.params["sample_every"]*len(self.train_dataloader))
			if ((step != 0) and (sample_every) and  ((step +1) % sample_every == 0)):
				# elapsed = format_time(time.time()-t0)
				# print(f'Batch {step} of {len(self.train_dataloader)}. Loss: {batch_loss}. Time: {elapsed}')
				self.sample()
				

			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

		avg_train_loss = total_train_loss / len(self.train_dataloader)
		training_time = format_time(time.time()-t0)
		print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
  
		return avg_train_loss, training_time



	def setup_training(self, data, parameters = {}):
		# update parameters
		self.params = update_param_dict(self.params, parameters)
		json.dump(self.params, open(self.output_dir + "params.json", "w"), indent = 4)
  
		# print info
		print("-"*50)
		print("Starting training using:")
		print(json.dumps(self.params, indent=4))
		print(f"Using {self.device} for training")
		
		# load data
		self.load_data(data)
  
		# Setting seeds to enable reproducible runs
		set_seed() 

		# Using AdamW optimizer with default parameters
		self.optimizer = AdamW(self.model.parameters(), **self.params["optimizer"])

		# Toatl training steps is the number of data points times the number of epochs
		self.total_training_steps = len(self.train_dataloader)*self.params["epochs"]

		# Setting a variable learning rate using scheduler
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_training_steps=self.total_training_steps, **self.params["scheduler"])

		print("-"*50)
		print()
		

	def train(self):
		total_t0 = time.time()

		training_stats = []

		for epoch_i in range(self.params["epochs"]):
			torch.cuda.empty_cache()
			print(f'Beginning epoch {epoch_i+1} of {self.params["epochs"]}')
		
			# train one epoch
			avg_train_loss, training_time = self.train_one_epoch()
   
			# validate
			avg_val_loss, val_time = self.validate()

			# Record all statistics from this epoch.
			training_stats.append(
				{
					'epoch': epoch_i + 1,
					'Training Loss': avg_train_loss,
					'Valid. Loss': avg_val_loss,
					'Training Time': training_time,
					'Validation Time': val_time
				}
			)
			json.dump(training_stats, open(self.output_dir + "training_stats.json", "w"), indent = 4)
   
			# save model
			if (self.params["save_every"] is not None) and ((epoch_i+1) % self.params["save_every"] == 0):
				self.save()
    
			print("------------------------------")
    
		print(f'Total training took {format_time(time.time()-total_t0)}')

		# save model
		self.save()

if __name__ == "__main__":
	from image_to_poem.language_model.gpt2 import GPT2Model
	
	# setup
	max_poems = None
	params = {
		"sample_every": 0.05,
     	"dataset" : {
          	"max_texts" : max_poems
        }, 
      	"save_every" : 1,
    }
	
	# init model and data
	lm_model = GPT2Model()
	data = KagglePoems("data/kaggle_poems/topics/", max_poems = max_poems)
	
	# train
	trainer = Trainer(lm_model, data = data.poems, train_params=params)
	trainer.train()