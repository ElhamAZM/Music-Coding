'''
model_archive.py
A file that contains neural network models.
You can also implement your own model here.
'''
import torch.nn as nn
from collections import OrderedDict


class Baseline(nn.Module):
	def __init__(self, hparams):
		super(Baseline, self).__init__()

		self.conv0 = nn.Sequential(OrderedDict([
			('conv0',nn.Conv1d(hparams.num_mels,40, kernel_size=8, stride=1, padding=0)), #change the output channel 
                        
			('batchnorm0',nn.BatchNorm1d(40)),
                        
                        
                        
			('relu0',nn.ReLU()),
			('maxpool0',nn.MaxPool1d(8, stride=8, padding=0))
                        
                        
                        
                        
		]))

		self.conv1 = nn.Sequential(OrderedDict([
			('conv1',nn.Conv1d(40, 40, kernel_size=8, stride=1, padding=0)), #change the output and input channel
                        
			('batchnorm1',nn.BatchNorm1d(40)),
                        
                      
                        
			('relu1',nn.ReLU()),
			('maxpool1',nn.MaxPool1d(8, stride=8, padding=0))
                        
                        
                        
		]))

		self.conv2 = nn.Sequential(OrderedDict([
			('conv2',nn.Conv1d(40, 49, kernel_size=4, stride=1, padding=0)), #change the output and input channel
                        
			('batchnorm2',nn.BatchNorm1d(49)),
                        
                       
                       
			('relu2',nn.ReLU()),
			('maxpool2',nn.MaxPool1d(4, stride=4, padding=0))
                        
                        
                        
		]))

            

                

		self.linear = nn.Linear(147, len(hparams.genres)) # change the fully connected layer 
               

	def forward(self, x):
		x = x.transpose(1, 2)
		x = self.conv0(x)
		x = self.conv1(x)
		x = self.conv2(x)
		
		
		

		x = x.view(x.size(0), x.size(1)*x.size(2))
		x = self.linear(x)

		return x

