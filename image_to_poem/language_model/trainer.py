import numpy as np
import matplotlib.pyplot as plt
import random
import json
import time
import datetime
import os
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

import torch
from torch.optim import AdamW

from image_to_poem.utils import format_time, update_param_dict
from image_to_poem.language_model.lm_dataset import get_datasets, create_dataloader
from image_to_poem.data.kaggle_poems import KagglePoems
from image_to_poem.utils import save_json_file


def set_seed(seed_val = 42):
	"""
	Sets the seed for random number generation for reproducibility.
	
	Parameters
	----------
	seed_val (int, optional):
		Seed for random number generation. Defaults to 42.
	"""
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

class Trainer:
	"""
	A class for training language models.
	"""
    # set default parameters
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
			"max_length": 500,
			"num_return_sequences":1,
		},
		"dataset" : {
			"train_frac": 0.9, 
   			"max_length": 500,
	  		"max_texts": None,
		},
	}
	
	def __init__(self, lm_model, data, train_params = {}, verbose = False):
		"""
		Initializes the trainer.
  
		Parameters
		----------
		lm_model (GPT2Model):
			An instance of the image_to_poem.language_model.gpt2.GPT2Model class. The name of this model will be used to create the output directory.
		data (list of str):
			List of texts to use for training.
		train_params (dict, optional):
			Dictionary of training parameters to update. Defaults to {}. 
			If you e.g. want to try a different learning rate, you can pass {"optimizer" : {"lr" : 1e-4}}.
			For dict of default parameters see the Trainer.params attribute.
		verbose (bool, optional):
			Whether to print verbose output. Defaults to False.   
  		"""
		self.verbose = verbose 
  
		# save lm model and extract necesary components
		self.device = lm_model.device
		self.tokenizer = lm_model.tokenizer
		self.model = lm_model.model
  
		# create output dir
		self.create_output_dir(lm_model)
  
		# create log dir
		self.log_dir = self.output_dir + "logs/"
		os.makedirs(self.log_dir, exist_ok = True)
  
		# setup training
		self.setup_training(data, train_params)
  
		# some tracking variables
		self.current_epoch = 0
		
  
	def create_output_dir(self, lm):
		"""
		Creates the output directory for the model.
  
		Parameters
		----------
		lm (GPT2Model):
			An instance of the image_to_poem.language_model.gpt2.GPT2Model class. The name of this model will be used to create the output directory.
  		"""
		# define output dir
		self.output_dir = "models/language_models/" + lm.name + "/"
  
		# append date if output dir already exists
		if os.path.exists(self.output_dir):
			print(f"WARNING: Output directory '{self.output_dir}' already exists. Appending the date to the file name...")
			self.output_dir = self.output_dir.strip("/")
			self.output_dir += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
   
		# create output dir
		os.makedirs(self.output_dir, exist_ok = True)

  
	def save(self):
		"""
		Saves the model and tokenizer to the output directory.
  		"""
		# create model dir
		model_dir = self.output_dir + "model/"
		print(f"Saving model to '{model_dir}' ...")
  
		# save model
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		model_to_save.save_pretrained(model_dir)
  
		# save tokenizer
		self.tokenizer.save_pretrained(model_dir)
		
	def sample(self):
		"""
		Creates samples from the model and saves them to the output directory.
		"""
		# create folder for samples
		sample_dir = self.output_dir + "samples/"
		os.makedirs(sample_dir, exist_ok = True)
  
		# turn on eval mode
		self.model.eval()

		# create samples
		sample_outputs = self.model.generate(bos_token_id=random.randint(1,30000), 
									   pad_token_id=self.tokenizer.pad_token_id,
									   **self.params["sample"])
  
		# decode and save samples
		for i, sample_output in enumerate(sample_outputs):
			sample_ = self.tokenizer.decode(sample_output, skip_special_tokens=True)
			with open(sample_dir + f"epoch_{self.current_epoch}_batch_{self.current_batch_no}_{i}.txt", "w", encoding="utf8") as file:
				file.write(sample_)

		# go back to train mode
		self.model.train()
  
	def load_data(self, data):
		"""
		Loads the data and creates the training and validation datasets.
  
		Parameters
		----------
		data (list of str):
			List of texts to use for training.
		"""
		self.train_dataset, self.val_dataset = get_datasets(data, self.tokenizer, **self.params["dataset"], verbose=self.verbose)
		self.train_dataloader = create_dataloader(self.train_dataset, train=True, batch_size=self.params["batch_size"], verbose=self.verbose)
		self.val_dataloader = create_dataloader(self.val_dataset, train=False, batch_size=self.params["batch_size"], verbose=self.verbose)

	def validate(self, verbose = False):
		"""
		Validates the model on the validation dataset.

		Parameters
		----------
		verbose (bool, optional):
			Whether to print verbose output. Defaults to False.
		"""
		# turn on eval mode
		self.model.eval()

		# init variables
		t0 = time.time()
		total_eval_loss = 0

		# go through batches
		for batch in self.val_dataloader:
			# get batch and move to device
			b_input_ids = batch[0].to(self.device)
			b_labels = batch[0].to(self.device)
			b_masks = batch[1].to(self.device)
				
			with torch.no_grad():   
				# Forward pass
				outputs  = self.model(b_input_ids,  
								attention_mask = b_masks,
								labels=b_labels)
				
				# get loss
				loss = outputs[0]  
				batch_loss = loss.item()
				total_eval_loss += batch_loss   

		# calculate stats
		avg_val_loss = total_eval_loss / len(self.val_dataloader)  
		val_time = format_time(time.time() - t0)    
		if verbose:
			print(f'Validation loss: {avg_val_loss}. Validation Time: {val_time}')
  
		return avg_val_loss, val_time

	def train_one_epoch(self):
		"""
		Trains the model for one epoch.
  		"""
		# reset variables
		self.current_epoch += 1
		self.current_batch_no = 0  
		t0 = time.time()
		total_train_loss = 0
  
		# turn on train mode
		self.model.train()

		# create progress bar
		pbar = tqdm(enumerate(self.train_dataloader), 
				desc = f"Training epoch {self.current_epoch} out of {self.params['epochs']}", 
			 	total = len(self.train_dataloader))

		# go through batches
		for step, batch in pbar:
			self.current_batch_no += 1
   
			# get batch and move to device
			b_input_ids = batch[0].to(self.device)
			b_labels = batch[0].to(self.device)
			b_masks = batch[1].to(self.device)

			# clear gradients
			self.model.zero_grad()

			# Forward pass
			outputs = self.model(b_input_ids,
							labels=b_labels,
							attention_mask=b_masks)
			
			# get loss
			loss = outputs[0]
			batch_loss = loss.item()
			total_train_loss += batch_loss
   
			# log batch
			self.log("batch_loss", batch_loss)

			# Sampling every x steps
			sample_every = sample_every if self.params["sample_every"] > 1 else int(self.params["sample_every"]*len(self.train_dataloader))
			if ((step != 0) and (sample_every) and  ((step +1) % sample_every == 0)):
				# sample
				self.sample()
	
				# validate and save current avg batch loss
				avg_val_loss, val_time = self.validate()
				self.log("val_loss", avg_val_loss)
				
			# Backward pass
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

		# calculate stats
		avg_train_loss = total_train_loss / len(self.train_dataloader)
		training_time = format_time(time.time()-t0)
		print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
  
		return avg_train_loss, training_time


	def log(self, log_file, value):
		"""
		Logs a value to a file.

		Parameters
		----------
		log_file (str):
			Name of the log file.
		value (float):
			Value to log.
		"""
		# create path
		path = self.log_dir + log_file + ".txt"
  
		# create file if it doesn't exist
		if not os.path.exists(path):
			with open(path, "w") as file:
				pass

		# append value to file
		with open(path, "a") as file:
			file.write(f"{value}\n")


	def setup_training(self, data, parameters = {}):
		"""
		Sets up the training.

		Parameters
		----------
		data (list of str):
			List of texts to use for training.
		parameters (dict, optional):
			Dictionary of training parameters to update. Defaults to {}. 
			If you e.g. want to try a different learning rate, you can pass {"optimizer" : {"lr" : 1e-4}}.
			For dict of default parameters see the Trainer.params attribute.
		"""
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
		"""
		Trains the model.
		"""
		# init variables
		total_t0 = time.time()
		training_stats = []

		# go through epochs
		for epoch_i in range(self.params["epochs"]):
			# empty cache
			torch.cuda.empty_cache()
   
			# print info
			print(f'Beginning epoch {epoch_i+1} of {self.params["epochs"]}')
		
			# train one epoch
			avg_train_loss, training_time = self.train_one_epoch()
   
			# validate
			avg_val_loss, val_time = self.validate(verbose=self.verbose)

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
			self.log("train_loss_epoch", avg_train_loss)
			self.log("val_loss_epoch", avg_val_loss)
   
			# save model
			if (self.params["save_every"] is not None) and ((epoch_i+1) % self.params["save_every"] == 0):
				self.save()
	
			print("------------------------------")
	
		print(f'Total training took {format_time(time.time()-total_t0)}')

		# save model
		self.save()
  
		# plot losses
		self.plot_losses()	
  
	def plot_losses(self):
		"""
		Plots the losses logged during training.
		"""
		# load losses
		load_loss = lambda log_file: np.loadtxt(self.log_dir + log_file + ".txt") if os.path.exists(self.log_dir + log_file + ".txt") else np.array([])
		val_loss = load_loss("val_loss")
		train_loss = load_loss("batch_loss")	
		val_loss_epoch = load_loss("val_loss_epoch")
		train_loss_epoch = load_loss("train_loss_epoch")
  
		# get some constants
		num_epochs = len(val_loss_epoch)
		len_train_loader = len(self.train_dataloader)
		num_samples = int(len_train_loader/int(len_train_loader*self.params["sample_every"]))

		# plot loss
		fig, ax = plt.subplots(figsize=(10, 5))
	
		ax.plot(np.arange(1, num_epochs+1), val_loss_epoch, label="Validation Loss")
		ax.plot(np.arange(1, num_epochs+1), train_loss_epoch, label="Training Loss")
		ax.plot(np.arange(1, num_epochs*num_samples+1)/num_samples, val_loss, label="Validation Loss (per sample)")
		ax.plot(np.arange(1, num_epochs*len_train_loader+1)/len_train_loader, train_loss, label="Training Loss (per batch)")

		ax.legend()
		ax.set(xlabel='Epoch', ylabel='Loss',title='Loss over training')

		plt.savefig(self.output_dir + "losses.pdf")

def save_class_settings(input_class, path, keys_to_exclude = []):
	"""
    Saves the settings of a class to a json file.
    
    Parameters
	----------
	input_class (object):
		An instance of a class.
	path (str):
		Path to save the json file to.
	keys_to_exclude (list of str, optional):
		List of keys to exclude from the settings. Defaults to [].
    """
    # get settings from the __dict__ attribute
	settings = input_class.__dict__.copy()
	
	# remove keys to exclude
	for key in keys_to_exclude:
		del settings[key]
  
	# convert values to strings
	for key in settings:
		if not str(settings[key]).isnumeric():
			settings[key] = str(settings[key])
	
	# save to json file
	save_json_file(path, settings)



if __name__ == "__main__":
	from image_to_poem.language_model.gpt2 import GPT2Model
	
	# setup
	max_poems = 100
	params = {
		"batch_size" : 4,
		"sample_every": 0.25,
	 	"dataset" : {
		  	"max_texts" : max_poems,
		   	"max_length" : 100,
		}, 
	  	"save_every" : 1,
	   "epochs" : 2,
	}
	
	# init model and data
	lm_model = GPT2Model(pretrained_model="gpt2", name = "test")
	data = KagglePoems("data/kaggle_poems/", max_poems = max_poems)
	
	# create trainer
	trainer = Trainer(lm_model, data = data.poems, train_params=params)
 
	# save data settings
	save_class_settings(data, trainer.output_dir + "data_settings.json", keys_to_exclude=["words", "poems"])

	# save model settings
	save_class_settings(lm_model, trainer.output_dir + "model_settings.json")

	# train 
	trainer.train()