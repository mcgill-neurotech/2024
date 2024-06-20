import torch
from torch import nn 
from torch.nn import functional as F
import lightning as L
import dataclasses
from typing import Tuple
from einops import reduce,rearrange, repeat
from torchsummary import summary
from pytorch_lightning.utilities.model_summary import ModelSummary
import math

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb"""


class EEGNet(L.LightningModule):
	def __init__(self,cond,d_out):
		super(EEGNet, self).__init__()
		self.T = 120
		
		# Layer 1
		self.conv1 = nn.Conv2d(1, 16, (cond, 64), padding = 0)
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
		return x
	
	def classify(self,x):
		x = rearrange(x,"b d t -> b 1 d t")
		x = self.forward(x)
		x = rearrange(x,"b d1 d2 t -> b (d1 d2 t)")
		x = self.out_proj(x)
		return x
	
class DecoderBlock(L.LightningModule):

	def __init__(self,
			  x_dim,
			  g_dim,
			  kernel_size,
			  class_dim,
			  time_dim) -> None:
		super().__init__()
		self.up_conv = nn.ConvTranspose1d(g_dim,x_dim,kernel_size=kernel_size,
									padding=kernel_size//2,stride=4)
		self.conv = nn.Conv1d(2*x_dim,x_dim,kernel_size=5,padding=5//2)
		self.embeds = Embed(class_dim,time_dim,x_dim)

	def forward(self,x,g,t,cond):

		g = self.up_conv(g)
		dtype = g.dtype
		g = F.upsample(g.to(torch.float32),x.shape[-1])
		g = g.to(dtype)
		g = self.embeds(g,t,cond)
		x = self.embeds(g,t,cond)
		x = F.elu(self.conv(torch.concat((x,g),1)))
		x = F.instance_norm(x)
		x = F.dropout(x,0.25)
		return x
	
@torch.jit.script
def crop_add(x:torch.Tensor,t:torch.Tensor,cond:torch.Tensor):
	t = t[...,0:x.shape[-1]]
	cond = cond[...,0:x.shape[-1]]
	return x + t + cond

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
	
class Embed(L.LightningModule):

	def __init__(self,class_dim,time_dim,out_channels) -> None:
		super().__init__()
		self.time_embed = nn.Sequential(
			nn.ELU(),
			nn.Conv1d(time_dim,out_channels,
			 kernel_size=1,stride=1,padding="same",
			 bias=True)
		)

		self.class_embed = nn.Sequential(
			nn.ELU(),
			nn.Conv1d(class_dim,out_channels,
			 kernel_size=1,stride=1,padding="same",
			 bias=True)
		)

	def forward(self,x,t,cond):

		t = self.time_embed(t)
		cond = self.class_embed(cond)

		return crop_add(x,t,cond)
	
class SinusoidalPosEmb(L.LightningModule):
    """
    Sinusoidal time embedding.
    """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.register_buffer("aranged", torch.arange(self.half_dim))

    def forward(self, x):
        emb = math.log(10000.0) / (self.half_dim - 1)
        emb = torch.exp(self.aranged * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
	
class EEGUNet(L.LightningModule):
	def __init__(self,
			  time_dim,
			  class_dim,
			  cond,
			  d_out):
		super(EEGUNet, self).__init__()
		self.T = 512
		self.signal_length = 512
		self.signal_channel = cond

		self.time_embed = SinusoidalPosEmb(time_dim)

		self.class_embed = nn.Embedding(3,class_dim)
		
		# Layer 1
		self.conv1 = nn.Conv2d(1, 16, (cond, 65), padding = (0,32))

		# Layer 2
		self.conv2 = nn.Conv2d(1, 4, (3, 33),padding=(1,16))
		self.pooling2 = nn.MaxPool2d(2, 4)
		self.embed2 = Embed(class_dim,time_dim,16)
		
		# Layer 3
		self.embed3 = Embed(class_dim,time_dim,4*8)
		self.conv3 = nn.Conv2d(4, 4, (9, 5),padding=(9//2,2))

		self.bottle_conv = nn.Conv1d(16,16,kernel_size=5,padding=2)
		self.pooling3 = nn.MaxPool2d((2, 4))
		self.out_proj = nn.Linear(d_out,2)

		self.decode1 = DecoderBlock(4*8,16,5,class_dim,time_dim)
		self.decode2 = DecoderBlock(4*16,4*8,5,class_dim,time_dim)

		self.out_conv = nn.Conv1d(4*16,1,1)
		
	def forward(self, x,t,cond):

		x = rearrange(x,"b d t -> b 1 d t")
		batch,_,dim,length = x.shape

		cond = cond[:,0,0]
		cond = self.class_embed(cond.to(torch.int))
		cond = repeat(cond,"b d -> b d t",t=length)

		t = self.time_embed(t)

		t = repeat(t,"b d -> b d l",l=length)

		# Layer 1 -> 16
		x = F.elu(self.conv1(x))
		x = F.instance_norm(x)
		x = F.dropout(x, 0.25)
		x = rearrange(x,"b cond h w ->b h cond w")

		# Layer 2 -> 4*16
		batch,channel,height,time = x.shape

		x = rearrange(x,"b cond h t -> b (cond h) t",h=height)
		x = self.embed2(x,t,cond)
		x = rearrange(x,"b (cond h) t  -> b cond h t",h=height)

		x = F.elu(self.conv2(x))
		x = F.instance_norm(x)
		skip_1 = F.dropout(x, 0.25)
		x = reduce(skip_1,"b cond (h k1) (t k2) -> b cond h t","max",k1=2,k2=4)
		
		# Layer 3 -> 4*8
		batch,channel,height,time = x.shape

		x = rearrange(x,"b cond h t -> b (cond h) t",h=height)
		x = self.embed3(x,t,cond)
		x = rearrange(x,"b (cond h) t  -> b cond h t",h=height)

		x = F.elu(self.conv3(x))
		x = F.instance_norm(x)
		skip_2 = F.dropout(x, 0.25)
		x = reduce(skip_2,"b cond (h k1) (t k2) -> b cond h t","max",k1=2,k2=4)

		skip_2 = rearrange(skip_2,"b d h t -> b (d h) t")
		skip_1 = rearrange(skip_1,"b d h t -> b (d h) t")

		x = rearrange(x,"b d h t -> b (d h) t")
		x = self.bottle_conv(x)
		x = self.decode1(skip_2,x,t,cond)
		x = self.decode2(skip_1,x,t,cond)
		x = self.out_conv(x)
		return x
	
	def conditional_forward(self,x,t,cond,w):

		n = len(x)
		x,t,cond = double_inputs(x,t,cond)
		x = self.forward(x,t,cond)
		x = dedouble_outputs(x,w)
		return x
	
	def classify(self,x):
		b,d,t = x.shape
		x = rearrange(x,"b d t -> b 1 d t")
		cond = torch.zeros((b,d,t)).to(self.device)
		t = torch.zeros(b).to(self.device)

		batch,_,dim,length = x.shape

		cond = cond[:,0,0]
		cond = self.class_embed(cond.to(torch.int))
		cond = repeat(cond,"b d -> b d t",t=length)

		t = self.time_embed(t)

		t = repeat(t,"b d -> b d l",l=length)

		# Layer 1 -> 16
		x = F.elu(self.conv1(x))
		x = F.instance_norm(x)
		x = F.dropout(x, 0.25)
		x = rearrange(x,"b cond h w ->b h cond w")

		# Layer 2 -> 4*16
		batch,channel,height,time = x.shape

		x = rearrange(x,"b cond h t -> b (cond h) t",h=height)
		x = self.embed2(x,t,cond)
		x = rearrange(x,"b (cond h) t  -> b cond h t",h=height)

		x = F.elu(self.conv2(x))
		x = F.instance_norm(x)
		x = F.dropout(x, 0.25)
		x = reduce(x,"b cond (h k1) (t k2) -> b cond h t","max",k1=2,k2=4)
		
		# Layer 3 -> 4*8
		batch,channel,height,time = x.shape

		x = rearrange(x,"b cond h t -> b (cond h) t",h=height)
		x = self.embed3(x,t,cond)
		x = rearrange(x,"b (cond h) t  -> b cond h t",h=height)

		x = F.elu(self.conv3(x))
		x = F.instance_norm(x)
		x = F.dropout(x, 0.25)
		x = reduce(x,"b cond (h k1) (t k2) -> b cond h t","max",k1=2,k2=4)
		print(x.shape)
		x = rearrange(x,"b cond h t -> b (cond h t)")
		x = self.out_proj(x)
		return x
	
if __name__ == "__main__":

	model = EEGUNet(8,8,2,512)

	print(ModelSummary(model))
	 
	model.to("cuda")

	x = torch.rand(2,1,2,512).to("cuda")
	t = torch.tensor([25,50]).to("cuda")
	cond = torch.tensor([0,1]).to("cuda")
	cond = repeat(cond,"b -> b cond t",cond=2,t=512)
	y = model(x,t,cond)
	print(y.shape)
	pred = model.classify(rearrange(x,"b 1 cond t -> b cond t"))
	print(pred.shape)