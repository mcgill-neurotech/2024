import torch
from torch import nn 
from torch.nn import functional as F
import lightning as L
import dataclasses
from typing import Tuple
from einops import reduce,rearrange
import torchsummary

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
	pool_op:nn.Module
	residual: bool = False
	p_drop: float

	def new_shapes(
				self,
				input_channels,
				ouput_channels):
		
		return ConvConfig(
			input_channels=input_channels,
			output_channels=ouput_channels,
			conv_op=self.conv_op,
			norm_op=self.norm_op,
			non_lin=self.non_lin,
			groups=self.groups,
			padding=self.padding,
			kernel_size=self.kernel_size,
			pool_fact=self.pool_fact,
			pool_op=self.pool_op,
			residual=self.residual,
			p_drop=self.p_drop
		)

@dataclasses.dataclass(kw_only=True)
class EncodeConfig(ConvConfig):
	input_channels: int
	output_channels: int
	conv_op: nn.Module
	norm_op: nn.Module
	non_lin: nn.Module
	groups: int = 1
	padding: Tuple[int, ...] = (1,1)
	kernel_size: Tuple[int, ...] = (3,3)
	pool_fact: int = 2
	p_drop: float

@dataclasses.dataclass(kw_only=True)
class DecodeConfig:
	x_channels: int
	g_channels: int 
	output_channels: int
	up_conv: nn.Module
	groups: int = 1
	padding: Tuple[int,...]
	kernel_size: Tuple[int,...] = (2,2)
	stride: Tuple[int,...] = (2,2)
	conv_config: ConvConfig

	def new_shapes(self,
				x_channels,
				g_channels,
				output_channels):
		
		new_conv_config = self.conv_config.new_shapes(2*x_channels,output_channels)
		return DecodeConfig(
			x_channels=x_channels,
			g_channels=g_channels,
			output_channels=output_channels,
			up_conv=self.up_conv,
			groups=self.groups,
			padding=self.padding,
			kernel_size=self.kernel_size,
			stride=self.stride,
			conv_config=new_conv_config
		)

@dataclasses.dataclass(kw_only=True)
class UnetConfig:
	input_shape: Tuple[int,...]
	input_channels: int
	conv_op: nn.Module
	norm_op: nn.Module
	non_lin: nn.Module
	pool_op: nn.Module
	up_op: nn.Module
	starting_channels: int
	max_channels: int
	conv_group: int
	conv_padding: Tuple[int,...]
	conv_kernel: Tuple[int,...]
	pool_fact: Tuple[int,...]
	deconv_group: int
	deconv_padding: Tuple[int,...]
	deconv_kernel: Tuple[int,...]
	deconv_stride: Tuple[int,...]
	residual: False
	conv_pdrop: float = 0.1
	mlp_pdrop: float = 0.25

@dataclasses.dataclass(kw_only=True)
class BottleneckClassifierConfig:
	channels = Tuple[int,...]
	pool_op = nn.Module


class Convdown(L.LightningModule):
	def __init__(self, config: ConvConfig) -> None:
		super().__init__()

		self.c1 = config.conv_op(config.input_channels, config.output_channels, kernel_size=config.kernel_size, padding=config.padding, groups=config.groups)
		self.c2 = config.conv_op(config.output_channels, config.output_channels, kernel_size=config.kernel_size, padding=config.padding, groups=config.groups)
		self.drop = nn.Dropout(config.p_drop)
		self.instance_norm = config.norm_op(config.output_channels)
		self.non_lin = config.non_lin()
		self.residual = config.residual

	def forward(self, x):

		if self.residual:
			x = self.drop(self.c1(x))
			x = self.non_lin(x)
			x = x + self.drop(self.c2(x))
			x = self.non_lin(x)
		else:
			x = self.drop(self.c1(x))
			x = self.non_lin(x)
			x = self.drop(self.c2(x))
			x = self.non_lin(x)
		x = self.instance_norm(x)
		return x
	
class SingleConvDown(L.LightningModule):

	def __init__(self, config) -> None:
		super().__init__()
		self.conv = config.conv_op(config.input_channels,config.output_channels,
							 kernel_size=config.kernel_size,padding=config.padding)
		self.norm = config.norm_op(config.output_channels)

class Encode(L.LightningModule):
	def __init__(self, config: ConvConfig) -> None:
		super().__init__()
		self.convdown = Convdown(config)
		self.pool = config.pool_op(config.pool_fact)

	def forward(self, x):
		x = self.convdown(x)
		pooled = self.pool(x)
		return pooled, x

class Decode(L.LightningModule):

	def __init__(self, config: DecodeConfig) -> None:
		super().__init__()
		# self.deconv = config.up_op(g_channel,x_channel,2,2)
		self.deconv = config.up_conv(config.g_channels,
							   config.x_channels,
							   config.kernel_size,
							   config.stride,
							   groups=config.groups)
		self.conv = Convdown(config.conv_config)

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

class BottleNeckClassifier(L.LightningModule):

	def __init__(self,
			  channels: Tuple[int],
			  pool=None) -> None:
		super().__init__()

		self.mlp = nn.ModuleList()
		for i in range(len(channels)-1):
			self.mlp.append(nn.Linear(channels[i],channels[i+1]))
			self.mlp.append(nn.ReLU())
		self.mlp.append(nn.Linear(channels[-1],2))
		self.pool = pool

	def forward(self,x):
		if self.pool=="max":
			x = reduce(x,"b c t -> b c","max")
		elif self.pool=="mean":
			x = reduce(x,"b c t -> b c","mean")
		else:
			x = rearrange(x,"b c t -> b (c t)")
		for i in self.mlp:
			x = i(x)
		return x
		
class Unet(L.LightningModule):

	"""
	base Unet model with adaptable topology and dimension in nnUnet style.
	
	Attributes:
		config: configuration for Unet
	
	"""

	def __init__(self,
				 config: UnetConfig,
				 classifier: L.LightningDataModule
				 ):
		
		super().__init__()

		self.input_features = [config.starting_channels]
		size = torch.tensor(config.input_shape)

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

		# can't divice 0-d tensor
		get_min = lambda x: min(x) if len(size.shape)>0 else x

		while (get_min(size) > 8) & (len(self.input_features)<=6):
			if 2*self.input_features[-1] < config.max_channels:
				self.input_features.append(2*self.input_features[-1])
			else:
				self.input_features.append(config.max_channels)
			size = size/config.pool_fact

		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()
		self.out_shape = [get_min(size),self.input_features[-1]]

		# input_channels = [32, 64, 128, 256,256....]

		self.auxiliary_clf = classifier

		self.base_conv_config = ConvConfig(
			input_channels=1,
			output_channels=1,
			conv_op=config.conv_op,
			norm_op=config.norm_op,
			non_lin=config.non_lin,
			groups=config.conv_group,
			padding=config.conv_padding,
			kernel_size=config.conv_kernel,
			pool_fact=config.pool_fact,
			pool_op=config.pool_op,
			residual=config.residual,
			p_drop=config.conv_pdrop
		)

		self.base_decode_config = DecodeConfig(
			x_channels=1,
			g_channels=1,
			output_channels=1,
			up_conv=config.up_op,
			groups=config.deconv_group,
			padding=config.deconv_padding,
			kernel_size=config.deconv_kernel,
			stride=config.deconv_stride,
			conv_config=self.base_conv_config
		)

		input_channels = config.input_channels

		for idx,i in enumerate(self.input_features[:-1]):
			encode_config = self.base_conv_config.new_shapes(input_channels,i)
			self.encoder.append(Encode(encode_config))
			input_channels = i

		bottleneck_conv_config = self.base_conv_config.new_shapes(
			input_channels=self.input_features[-2],
			ouput_channels=self.input_features[-1]
		)

		self.middle_conv = Convdown(bottleneck_conv_config)

		output_features = self.input_features[::-1]

		for i in range(len(self.input_features)-1):

			decode_config = self.base_decode_config.new_shapes(
				x_channels=output_features[i+1],
				g_channels=output_features[i],
				output_channels=output_features[i+1]
			)

			self.decoder.append(Decode(decode_config))

		self.output_conv = config.conv_op(config.starting_channels,config.input_channels,1)

	def forward(self,
			 x):

		"""
		Full U-net forward pass to get the reconstructed datas
		"""
		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x)
			skip_connections.append(skip)

		x = self.middle_conv(x)

		for decode,skip in zip(self.decoder,reversed(skip_connections)):
			x = decode(skip,x)

		x = self.output_conv(x)
		return x
	
	def classify(self,x):
		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x)
			skip_connections.append(skip)

		x = self.middle_conv(x)
		y = self.auxiliary_clf(x)

		return y
			
	
Unet2D = UnetConfig(
	input_shape=(256,256),
	input_channels=1,
	conv_op=nn.Conv2d,
	norm_op=nn.InstanceNorm2d,
	non_lin=nn.ReLU,
	pool_op=nn.AvgPool2d,
	up_op=nn.ConvTranspose2d,
	starting_channels=32,
	max_channels=256,
	conv_group=1,
	conv_padding=(3,3),
	conv_kernel=(7,7),
	pool_fact=2,
	deconv_group=1,
	deconv_padding=(0,0),
	deconv_kernel=(2,2),
	deconv_stride=(2,2),
	residual=True
)

Chenetal2021 = UnetConfig(
	input_shape=(64,64),
	input_channels=3,
	conv_op=nn.Conv2d,
	norm_op=nn.InstanceNorm2d,
	non_lin=nn.ReLU,
	pool_op=nn.AvgPool2d,
	up_op=nn.ConvTranspose2d,
	starting_channels=64,
	max_channels=64,
	conv_group=1,
	conv_padding=(1,1),
	conv_kernel=(3,3),
	pool_fact=2,
	deconv_group=1,
	deconv_padding=(0,0),
	deconv_kernel=(2,2),
	deconv_stride=(2,2),
	residual=True
)

Unet1D = UnetConfig(
	input_shape=(256),
	input_channels=3,
	conv_op=nn.Conv1d,
	norm_op=nn.InstanceNorm1d,
	non_lin=nn.ReLU,
	pool_op=nn.AvgPool1d,
	up_op=nn.ConvTranspose1d,
	starting_channels=32,
	max_channels=256,
	conv_group=1,
	conv_padding=(1),
	conv_kernel=(3),
	pool_fact=2,
	deconv_group=1,
	deconv_padding=(0),
	deconv_kernel=(2),
	deconv_stride=(2),
	residual=True
)
	
if __name__ == "__main__":
	
	classifier = BottleNeckClassifier((4096,512))
	unet_2d = Unet(Chenetal2021,classifier)
	torchsummary.summary(unet_2d,(3,64,64),1,device="cpu")
	# x = torch.rand((4,3,64,64))
	# with torch.no_grad():
	# 	y = unet_2d.classify(x)
	# 	print(y)