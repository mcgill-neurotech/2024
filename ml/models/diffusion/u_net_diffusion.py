"""
Classifier free guidance based on: 
https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch
"""

import torch
from torch import nn 
from torch.nn import functional as F
import lightning as L
import dataclasses
from typing import Tuple
from einops import reduce,rearrange
import torchsummary
import sys
from pytorch_lightning.utilities.model_summary import ModelSummary
from einops import repeat,rearrange
sys.path.append("../../../motor-imagery-classification-2024/")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ntd.networks import SinusoidalPosEmb
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from models.unet.eeg_unets import (ConvConfig,
								   EncodeConfig,
								   DecodeConfig,
								   UnetConfig,
								   Convdown,
								   Encode,
								   Decode,
								   Unet,
								   BottleNeckClassifier)

@dataclasses.dataclass(kw_only=True)
class EmbedDecodeConfig(DecodeConfig):
	time_dim: int
	class_dim: int
	num_classes: int
	embed_act: nn.Module

	def new_shapes(self,
				x_channels,
				g_channels,
				output_channels):
		
		new_conv_config = self.conv_config.new_shapes(2*x_channels,output_channels)
		return EmbedDecodeConfig(
			time_dim = self.time_dim,
			class_dim = self.class_dim,
			num_classes = self.num_classes,
			embed_act = self.embed_act,
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
class EmbedConvConfig(ConvConfig):
	time_dim: int
	class_dim: int
	num_classes: int
	embed_act: nn.Module

	def new_shapes(
				self,
				input_channels,
				ouput_channels):
		
		return EmbedConvConfig(
			time_dim = self.time_dim,
			class_dim = self.class_dim,
			num_classes = self.num_classes,
			embed_act = self.embed_act,
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
class DiffusionUnetConfig(UnetConfig):
	time_dim: int
	class_dim: int
	num_classes: int

@torch.jit.script
def crop_add(x:torch.Tensor,t:torch.Tensor,c:torch.Tensor):
	t = t[...,0:x.shape[-1]]
	c = c[...,0:x.shape[-1]]
	return x + t + c

@torch.jit.script
def double_inputs(x:torch.Tensor,
				  t:torch.Tensor,
				  cond:torch.Tensor):
	
	x = torch.cat([x,x],0)
	t = torch.cat([t,t],0)
	cond = torch.cat([cond,0*cond],0)
	return x,t,cond

@torch.jit.script
def dedouble_outputs(x:torch.Tensor,
					 w:float):
	
	conditional = x[0:len(x)//2]
	unconditinoal = x[len(x)//2:]
	return (1+w)*conditional-w*unconditinoal

class EmbedConvdown(Convdown):

	def __init__(self, config: EmbedConvConfig) -> None:
		super().__init__(config)
		self.time_embed = nn.Sequential(
			config.embed_act(),
			nn.Conv1d(config.time_dim,config.output_channels,
			 kernel_size=1,stride=1,padding="same",
			 bias=True)
		)
		self.class_embed = nn.Sequential(
			config.embed_act(),
			nn.Conv1d(config.class_dim,config.output_channels,
			 kernel_size=1,stride=1,padding="same",
			 bias=True)
		)

	def forward(self, x,t,c):

		t = self.time_embed(t)
		c = self.class_embed(c)

		if self.residual:
			x = self.drop(self.c1(x))
			x = self.non_lin(x)
			x = crop_add(x,t,c)
			x = x + self.drop(self.c2(x))
			x = self.non_lin(x)
		else:
			x = self.drop(self.c1(x))
			x = self.non_lin(x)
			x = crop_add(x,t,c)
			x = self.drop(self.c2(x))
			x = self.non_lin(x)
		x = self.instance_norm(x)
		return x
	
class EmbedEncode(L.LightningModule):

	def __init__(self, config: EmbedConvConfig) -> None:
		super().__init__()
		self.convdown = EmbedConvdown(config)
		self.pool = config.pool_op(config.pool_fact)

	def forward(self, x,t,c):
		x = self.convdown(x,t,c)
		pooled = self.pool(x)
		return pooled, x
	
class EmbedDecode(L.LightningModule):

	def __init__(self, config: EmbedDecodeConfig) -> None:
		super().__init__()

		self.deconv = config.up_conv(config.g_channels,
							   config.x_channels,
							   config.kernel_size,
							   config.stride,
							   groups=config.groups)
		self.conv = EmbedConvdown(config.conv_config)

	def forward(self,x,g,t,c):
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
		x = self.conv(x,t,c)
		return x

class DiffusionUnet(L.LightningModule):

	"""
	base Unet model with adaptable topology and dimension in nnUnet style.
	
	Attributes:
		config: configuration for Unet
	
	"""

	def __init__(self,
				 config: DiffusionUnetConfig,
				 classifier: L.LightningDataModule
				 ):
		
		super().__init__()

		self.time_embed = SinusoidalPosEmb(config.time_dim)

		self.signal_length = config.input_shape
		self.signal_channel = config.input_channels

		# adding one for the null embedding
		self.class_embed = nn.Embedding(config.num_classes+1,
								  embedding_dim=config.class_dim)

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

		self.base_conv_config = EmbedConvConfig(
			embed_act=config.non_lin,
			time_dim=config.time_dim,
			class_dim=config.class_dim,
			num_classes=config.num_classes,
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

		self.base_decode_config = EmbedDecodeConfig(
			embed_act=config.non_lin,
			time_dim=config.time_dim,
			class_dim=config.class_dim,
			num_classes=config.num_classes,
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
			self.encoder.append(EmbedEncode(encode_config))
			input_channels = i

		bottleneck_conv_config = self.base_conv_config.new_shapes(
			input_channels=self.input_features[-2],
			ouput_channels=self.input_features[-1]
		)

		self.middle_conv = EmbedConvdown(bottleneck_conv_config)

		output_features = self.input_features[::-1]

		for i in range(len(self.input_features)-1):

			decode_config = self.base_decode_config.new_shapes(
				x_channels=output_features[i+1],
				g_channels=output_features[i],
				output_channels=output_features[i+1]
			)

			self.decoder.append(EmbedDecode(decode_config))

		self.output_conv = config.conv_op(config.starting_channels,config.input_channels,1)

	def forward(self,
			 x,
			 t,
			 cond):

		"""
		Full U-net forward pass to get the reconstructed datas
		"""

		# to match ntd, c is of shape batch x 1 x 
		
		c = cond

		batch,dim,length = x.shape

		c = c[:,0,0]
		c = self.class_embed(c.to(torch.int))
		c = repeat(c,"b d -> b d t",t=length)

		t = self.time_embed(t)

		t = repeat(t,"b d -> b d l",l=length)

		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x,t,c)
			skip_connections.append(skip)

		x = self.middle_conv(x,t,c)

		for decode,skip in zip(self.decoder,reversed(skip_connections)):
			x = decode(skip,x,t,c)

		x = self.output_conv(x)
		return x
	
	def conditional_forward(self,x,t,cond,w):

		n = len(x)
		x,t,c = double_inputs(x,t,cond)
		x = self.forward(x,t,c)
		x = dedouble_outputs(x,w)
		return x
	
	def classify(self,x):
		t = torch.zeros(len(x),device=x.device)
		t = self.time_embed(t)
		t = repeat(t,"b d -> b d l",l=x.shape[-1])

		c = torch.zeros((len(x)),device=x.device)
		c = self.class_embed(c.to(torch.int))
		c = repeat(c,"b d -> b d l",l=x.shape[-1])
		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x,t,c)
			skip_connections.append(skip)

		x = self.middle_conv(x,t,c)
		y = self.auxiliary_clf(x)

		return y
	
UnetDiff1D = DiffusionUnetConfig(
	time_dim=12,
	class_dim=12,
	num_classes=2,
	input_shape=(512),
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

	classifier = BottleNeckClassifier((2048,1024),)
	unet = DiffusionUnet(UnetDiff1D,classifier)
	print(ModelSummary(unet))
	x = torch.rand((2,3,512))
	t = torch.arange(0,2)
	c = repeat(torch.Tensor([0,1]),"b -> b 1 l",l=512).to(torch.long)
	print(unet.classify(x))