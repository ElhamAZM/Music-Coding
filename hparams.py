'''
hparams.py
A file sets hyper parameters for feature extraction and training.
You can change parameters using argument.
For example:
 $ python train_test.py --device=1 --batch_size=32.
'''
import argparse
import torch

class HParams(object):
	def __init__(self):
		# Dataset Settings
		self.dataset_path = './gtzan'
		self.feature_path = './feature'
		self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

		# Feature Parameters
		self.sample_rate = 22050
		self.fft_size = 1024
		self.win_size = 1024
		self.hop_size = 512
		self.num_mels = 40
		self.feature_length = 1200  # audio length = feature_length*hop_size/sample_rate (s) # change the length of the feature 

		# Training Parameters
		self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
		self.batch_size = 40  #change the batch size 
		self.num_epochs = 500 #change the number of epoch 
		self.learning_rate = 0.01 #change learning rate 
		self.stopping_rate =0.00001 #change the stopping rate 
		self.weight_decay = 0.00001 #change the weight decay 
		self.momentum = 0.9
		self.factor = 0.5 #change the factor 
		self.patience = 3
		
		

	# Function for parsing argument and set hyper parameters
	def parse_argument(self, print_argument=True):
		parser = argparse.ArgumentParser()
		for var in vars(self):
			value = getattr(hparams, var)
			argument = '--' + var
			parser.add_argument(argument, type=type(value), default=value)

		args = parser.parse_args()
		for var in vars(self):
			setattr(hparams, var, getattr(args, var))

		if print_argument:
			print('-------------------------')
			print('Hyper Parameter Settings')
			print('-------------------------')
			for var in vars(self):
				value = getattr(hparams, var)
				print(var + ': ' + str(value))
			print('-------------------------')

hparams = HParams()
hparams.parse_argument()


