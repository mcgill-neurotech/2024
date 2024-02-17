import torch
from torch import nn 
from torch.nn import functional as F
from torchsummary import summary

class Convdown(nn.Module):

	def __init__(self,
				 input_channels,
				 output_channels,
				 conv_op=nn.Conv3d,
				 norm_op=nn.InstanceNorm3d,
				 non_lin=nn.LeakyReLU,
				 kernel_size=3,
				 groups=1) -> None:
		super().__init__()
		# 3 x 3
		self.c1 = conv_op(input_channels,output_channels,kernel_size=kernel_size,padding=(kernel_size-1)//2,groups=groups)
		self.c2 = conv_op(output_channels,output_channels,kernel_size=kernel_size,padding=(kernel_size-1)//2,groups=groups)
		self.instance_norm = norm_op(output_channels)
		self.non_lin = non_lin()
		
	def forward(self,x):
		x = self.c1(x)
		x = self.non_lin(x)
		x = self.c2(x)
		x = self.non_lin(x)
		x = self.instance_norm(x)
		return x
	
class Encode(nn.Module):
	def __init__(self,
				 input_channels,
				 output_channels,
				 conv_op=nn.Conv3d,
				 norm_op=nn.InstanceNorm3d,
				 non_lin = nn.LeakyReLU,
				 pool_op = nn.MaxPool3d,
				 pool_fact = 2,
				 conv_kernel_size=3,
				 groups=1) -> None:
		super().__init__()
		self.convdown = Convdown(input_channels,output_channels,conv_op,norm_op,
								 non_lin,conv_kernel_size,groups)
		self.pool = pool_op(pool_fact)

	def forward(self,x):
		x = self.convdown(x)
		pooled = self.pool(x)
		return pooled,x
	
class Decode(nn.Module):

	def __init__(self,
				 x_channel,
				 g_channel,
				 output_channels,
				 up_op = nn.ConvTranspose3d,
				 conv_op = nn.Conv3d,
				 norm_op = nn.InstanceNorm3d,
				 non_lin = nn.LeakyReLU,
				 conv_kernel_size = 3,
				 ) -> None:
		super().__init__()
		self.deconv = up_op(g_channel,x_channel,2,2)
		self.conv = Convdown(2*x_channel,output_channels,conv_op,norm_op,non_lin,
							 conv_kernel_size)

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
	
class Unet(nn.Module):

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

		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		# input_channels = [32, 64, 128, 256,256....]

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

		for decode,skip in zip(self.decoder,reversed(skip_connections)):
			x = decode(skip,x)

		x = self.output_conv(x)
		return x