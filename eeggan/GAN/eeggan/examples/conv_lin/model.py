#import braindecode
from torch import nn
from eeggan.GAN.eeggan.modules.layers.reshape import Reshape,PixelShuffle2d
from eeggan.GAN.eeggan.modules.layers.normalization import PixelNorm
from eeggan.GAN.eeggan.modules.layers.weight_scaling import weight_scale
from eeggan.GAN.eeggan.modules.layers.upsampling import CubicUpsampling1d,CubicUpsampling2d,upsample_layer
from eeggan.GAN.eeggan.modules.layers.stdmap import StdMap1d
# from eeggan.modules.layers.fouriermap import FFTMap1d
from eeggan.GAN.eeggan.modules.progressive import ProgressiveGenerator,ProgressiveGeneratorBlock,\
							ProgressiveDiscriminator,ProgressiveDiscriminatorBlock
from eeggan.GAN.eeggan.modules.wgan import WGAN_I_Generator,WGAN_I_Discriminator
from torch.nn.init import calculate_gain
# from eeggan.modules.layers.xray import xrayscanner
# from eeggan.modules.layers.add_random_layer import add_random_layer
import numpy as np

n_featuremaps = 25

#base = starting samples => base = input_size/(2**N_blocks)
base = int(1016/(2**3))

Align = False

def create_disc_blocks(n_chans,base):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,out_filters,21,padding=21//2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								weight_scale(nn.Conv1d(out_filters,out_filters,9,padding=9//2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),

								weight_scale(nn.Conv1d(out_filters,out_filters,2,stride=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2))

	def create_in_sequence(n_chans,out_filters):
		return nn.Sequential(weight_scale(nn.Conv2d(1,out_filters,(101,3),padding=(50,1)),gain=calculate_gain('leaky_relu')),nn.LeakyReLU(0.2),
								weight_scale(nn.Conv2d(out_filters,out_filters,(1,n_chans)),
														gain=calculate_gain('leaky_relu')),
														
								Reshape([[0],[1],[2]]),
								nn.LeakyReLU(0.2))
	def create_fade_sequence(factor):
		return nn.AvgPool2d((factor,1),stride=(factor,1))
		
	blocks = []

	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence(n_featuremaps,n_featuremaps),
							create_in_sequence(n_chans,n_featuremaps),
							create_fade_sequence(2)



							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)

	tmp_block = ProgressiveDiscriminatorBlock(
							  nn.Sequential(StdMap1d(),
											create_conv_sequence(n_featuremaps+1,n_featuremaps),
											Reshape([[0],-1]),
											weight_scale(nn.Linear((n_featuremaps)*base,1),
															gain=calculate_gain('linear'))),
							  create_in_sequence(n_chans,n_featuremaps),
							  None
							  )
	blocks.append(tmp_block)
	return blocks

def create_gen_blocks(n_chans,z_vars):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(upsample_layer(mode='linear',scale_factor=2),  # 对3维张量只会scale最后一维
								weight_scale(nn.Conv1d(in_filters,out_filters,21,padding=21//2),
														gain=calculate_gain('leaky_relu')),
														
								nn.LeakyReLU(0.2),
								PixelNorm(),
								weight_scale(nn.Conv1d(out_filters,out_filters,9,padding=9//2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								PixelNorm()
								)

	def create_out_sequence(n_chans,in_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,n_chans,1),
														gain=calculate_gain('linear')),
								Reshape([[0],[1],[2],1]),
								PixelShuffle2d([1,n_chans]))  # (n_batch, 1, time, n_chan)
	def create_fade_sequence(factor):
		return upsample_layer(mode='bilinear',scale_factor=(2,1))  # 放大，为下层fade out做准备
	blocks = []

	tmp_block = ProgressiveGeneratorBlock(
								nn.Sequential(weight_scale(nn.Linear(z_vars,base*(n_featuremaps)),  # 代替1维卷积，改变时间维度
														gain=calculate_gain('leaky_relu')),
												nn.LeakyReLU(0.2),
												
												Reshape([[0],n_featuremaps,-1]),  # 时间应该还是最后一维
												create_conv_sequence(n_featuremaps,n_featuremaps)),  # 在chan维度上1维卷积，并两倍时间维
								create_out_sequence(n_chans,n_featuremaps),  # 回归到chan原始维度，相当于PGGAN的rgb层
								create_fade_sequence(2)  # 时间维度放大2倍，chan不变
								)
	
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								None
								)
	blocks.append(tmp_block)  # 时间维度只有插值？？？好吧可以理解，留意一下长度怎么对齐的
	return blocks


class Generator(WGAN_I_Generator):
	def __init__(self,n_chans,z_vars):
		super(Generator,self).__init__()
		self.model = ProgressiveGenerator(create_gen_blocks(n_chans,z_vars),conditional=False)

	def forward(self,input):
		return self.model(input)

class Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans,base),conditional=False)

	def forward(self,input):
		return self.model(input)

class Fourier_Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(Fourier_Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans,int(base/2)),conditional=False)

	def forward(self,input):
		return self.model(input)
