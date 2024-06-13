import lightning
from torch import nn
from typing import Tuple
from einops import rearrange,repeat
import torch.nn.functional as F
import torch

class DiffusionClf(lightning.LightningModule):

	def __init__(self,
			  model,
			  clf,
			  freeze=True):

		super().__init__()
		self.model = model.network
		if freeze:
			for param in self.model.parameters():
				param.requires_grad = False
		self.clf = clf

	def forward(self,
			 x):

		cond = torch.ones((x.shape[0],1,x.shape[-1]),device=self.device)
		x = torch.cat([x,cond],1)
		t = torch.zeros(len(x),device=self.device)
		time_embed = self.model.time_embbeder(t)
		time_embed = repeat(time_embed,"b t -> b t l",l=x.shape[-1])
		x = torch.cat([x,time_embed],1)
		x = self.model.conv_pool[0:-1](x)
		return x
	
	def classify(self,x):

		x = self.forward(x)
		x = self.clf(x)
		return x

class ClassificationHead(lightning.LightningModule):

	def __init__(self,
			  out_channels: Tuple[int],
			  pool=None,
			  **kwargs) -> None:
		super().__init__()

		self.mlp = nn.ModuleList()
		for i in range(len(out_channels)-1):
			self.mlp.append(nn.Linear(out_channels[i],out_channels[i+1]))
			self.mlp.append(nn.ReLU())
		self.mlp.append(nn.Linear(out_channels[-1],2))
		self.pool = pool

	def forward(self,x):
		x = x[...,-1]
		for i in self.mlp:
			x = i(x)
		return x
	
class EEGNetHead(lightning.LightningModule):
	def __init__(self,c_in,d_out,**kwargs):
		super().__init__()
		self.T = 120
		
		# Layer 1
		self.conv1 = nn.Conv2d(1, 16, (c_in, 1), padding = 0)
		self.norm1 = nn.BatchNorm2d(16, False)
		
		# Layer 2
		self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
		self.conv2 = nn.Conv2d(1, 4, (2, 32))
		self.norm2 = nn.BatchNorm2d(4, False)
		self.pooling2 = nn.MaxPool2d(2, 4)
		
		# Layer 3
		self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
		self.conv3 = nn.Conv2d(4, 4, (8, 4))
		self.norm3 = nn.BatchNorm2d(4, False)
		self.pooling3 = nn.MaxPool2d((2, 4))
		self.out_proj = nn.Linear(d_out,2)

	def forward(self, x):

		x = rearrange(x,"b d t -> b 1 d t")
		
		# Layer 1
		x = F.elu(self.conv1(x))
		x = self.norm1(x)
		x = F.dropout(x, 0.25)
		x = rearrange(x,"b cond h w ->b h cond w")

		# Layer 2
		x = self.padding1(x)
		x = F.elu(self.conv2(x))
		x = self.norm2(x)
		x = F.dropout(x, 0.25)
		x = self.pooling2(x)
		
		
		# Layer 3
		x = self.padding2(x)
		x = F.elu(self.conv3(x))
		x = self.norm3(x)
		x = F.dropout(x, 0.25)
		x = self.pooling3(x)

		x = rearrange(x,"b d1 d2 t -> b (d1 d2 t)")
		x = self.out_proj(x)
		return x
	
	def classify(self,x):
		return self.forward(x)