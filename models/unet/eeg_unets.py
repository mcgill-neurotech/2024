import torch
from torch import nn 
from torch.nn import functional as F
import lightning as L
import dataclasses
from typing import Tuple

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@dataclasses.dataclass(kw_only=True)
class ConvConfig:
	input_channels: int
	output_channels: int
	conv_op: nn.Module
	norm_op: nn.Module
	non_lin: nn.Module
	groups: int = 1
	padding: Tuple[int,...]
	kernel_size: Tuple[int,...]
	pool_fact: int

@dataclasses.dataclass(kw_only=True)
class ConvConfig:
    input_channels: int
    output_channels: int
    conv_op: nn.Module
    norm_op: nn.Module
    non_lin: nn.Module
    groups: int  # Added groups parameter
    padding: Tuple[int, ...]
    kernel_size: Tuple[int, ...]
    pool_fact: int  # Added pooling factor

    def __init__(self, input_channels: int, output_channels: int, conv_op: nn.Module,
                 norm_op: nn.Module, non_lin: nn.Module, groups: int = 1,
                 padding: Tuple[int, ...] = (0, 0), kernel_size: Tuple[int, ...] = (3, 3),
                 pool_fact: int = 2) -> None:
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.non_lin = non_lin
        self.groups = groups
        self.padding = padding
        self.kernel_size = kernel_size
        self.pool_fact = pool_fact

class Convdown(L.LightningModule):
    def __init__(self, config: ConvConfig) -> None:
        super().__init__()

        self.c1 = config.conv_op(config.input_channels, config.output_channels, kernel_size=config.kernel_size, padding=config.padding, groups=config.groups)
        self.c2 = config.conv_op(config.output_channels, config.output_channels, kernel_size=config.kernel_size, padding=config.padding, groups=config.groups)
        self.instance_norm = config.norm_op(config.output_channels)
        self.non_lin = config.non_lin()

    def forward(self, x):
        x = self.c1(x)
        x = self.non_lin(x)
        x = self.c2(x)
        x = self.non_lin(x)
        x = self.instance_norm(x)
        return x

class Encode(L.LightningModule):
    def __init__(self, input_channels, output_channels, config: ConvConfig) -> None:
        super().__init__()
        self.convdown = Convdown(config)
        self.pool = config.pool_op(config.pool_fact)

    def forward(self, x):
        x = self.convdown(x)
        pooled = self.pool(x)
        return pooled, x

class Decode(L.LightningModule):

	def __init__(self,
				 x_channel,
				 g_channel,
				 output_channels,
				 config: ConvConfig) -> None:
		super().__init__()
		self.deconv = config.up_op(g_channel,x_channel,2,2)
		self.conv = Convdown(2*x_channel,output_channels,config)

	def forward(self,x,g):
		"""
		x: high resolution image from the encoder stage
		g: low resolution representation from the decoder stage
		"""

		# g: N x 64 x 64
		# x: N x 32 x 128
		# deconv(g): N x 32 x 128
		# concat(g,x): N x 64 x 128
		# conv(concat(g,x)): N x 32 x 128

		g = self.deconv(g)
		x = torch.concat((x,g),1)
		x = self.conv(x)
		return x
	
class Unet(L.LightningModule):

	def __init__(self,
				 num_modalities,
				 n_dim,
				 starting_channels,
				 max_channels,
				 size,
				 groups=1):
		
		super().__init__()

		self.input_features = [starting_channels]
		input_channels = num_modalities

		"""
		possible input shapes:
		1. N x D x L
			starting with N examples D=2 (2 electrodes) with length L (time)
			going to N x 32 x L after 1st layer
			
		2. N X D x C x L
			starting with N examples D = 1 (one feature) with 2 electrodes with lenght L
			going to N x 32 x 2 x L

		3. N x D x C x F x L
			N examples D =1 feature 2 electrodes x frequency x L

		4. N x D x F x L
			we don't do the distinction between channels and features but add a frequency dimension
		"""

		if n_dim == 1:
			ops = {"conv_op":nn.Conv1d,
				   "norm_op":nn.InstanceNorm1d,
				   "non_lin":nn.LeakyReLU,
				   "pool_op":nn.MaxPool1d,
				   "up_op":nn.ConvTranspose1d}

		if n_dim == 2:
			ops = {"conv_op":nn.Conv2d,
				   "norm_op":nn.InstanceNorm2d,
				   "non_lin":nn.LeakyReLU,
				   "pool_op":nn.MaxPool2d,
				   "up_op":nn.ConvTranspose2d}
			
		elif n_dim == 3:
			ops = {"conv_op":nn.Conv3d,
				   "norm_op":nn.InstanceNorm3d,
				   "non_lin":nn.LeakyReLU,
				   "pool_op":nn.MaxPool3d,
				   "up_op":nn.ConvTranspose3d}

		while (size > 8) & (len(self.input_features)<=6):
			if 2*self.input_features[-1] < max_channels:
				self.input_features.append(2*self.input_features[-1])
			else:
				self.input_features.append(max_channels)
			size /= 2

		self.encoder = L.LightningModuleList()
		self.decoder = L.LightningModuleList()

		# input_channels = [32, 64, 128, 256,256....]

		self.auxiliary_clf = nn.Identity()

		for idx,i in enumerate(self.input_features[:-1]):
			self.encoder.append(Encode(input_channels,i,ops["conv_op"],ops["norm_op"],
									ops["non_lin"],ops["pool_op"],groups=groups))
			input_channels = i

		self.middle_conv = Convdown(self.input_features[-2],self.input_features[-1],ops["conv_op"],
									ops["norm_op"],ops["non_lin"])

		output_features = self.input_features[::-1]

		for i in range(len(self.input_features)-1):
			self.decoder.append(Decode(output_features[i+1],output_features[i],output_features[i+1],
									   ops["up_op"],ops["conv_op"],ops["norm_op"],
									   ops["non_lin"]))

		self.output_conv = ops["conv_op"](starting_channels,2,1)

		# N x 32 X L -> N x 2 x L

	def forward(self,x):
		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x)
			skip_connections.append(skip)

		x = self.middle_conv(x)
		y = self.auxiliary_clf(x)

		for decode,skip in zip(self.decoder,reversed(skip_connections)):
			x = decode(skip,x)

		x = self.output_conv(x)
		return x,y
	
if __name__ == "__main__":
	config = ConvConfig(
		input_channels=1,
		output_channels=32,
		conv_op=nn.Conv2d,
		norm_op=nn.InstanceNorm2d,
		non_lin=nn.LeakyReLU,
		groups=1,
		kernel_size=(3,3),
		padding= (0,0),
  		pool_fact=2
	)

	conv = Convdown(config)