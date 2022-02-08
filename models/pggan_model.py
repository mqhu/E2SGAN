import torch
from .base_model import BaseModel
from . import networks
import os
import sys
from gansynth.normalizer import DataNormalizer
from gansynth import PGGAN


class PGGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='eeg')
        # if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        #parser.add_argument('--loss_freq', type=int, default=50, help='frequency of saving loss plots')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake', 'Wasserstein', 'D', 'G', 'G_L1', 'D_grad']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['fake_B', 'real_B']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.latent_size_ = opt.input_nc
        is_sigmoid = False
        self.is_tanh_ = True
        self.avg_layer = torch.nn.AvgPool2d((2, 2), stride=(2, 2))
        self.AtoB = self.opt.direction == 'AtoB'

        # define networks (both generator and discriminator)
        netG = PGGAN.Generator(16, self.latent_size_, opt.output_nc,
                                is_tanh=self.is_tanh_, channel_list=[128, 128, 64, 32, 16])

        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            # netG.to(self.gpu_ids[0])
            # self.netG = torch.nn.DataParallel(netG, self.gpu_ids)  # PGGAN应该自带init weights功能
            self.netG = networks.init_net(opt, netG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, device=self.device)

        if self.isTrain:
            self.is_sigmoid_ = False if opt.gan_mode in ["wgangp"] else is_sigmoid
            netD = PGGAN.Discriminator(16, opt.input_nc, is_sigmoid=self.is_sigmoid_,
                                       channel_list=[128, 128, 64, 32, 16])
            # netD.to(self.gpu_ids[0])
            # self.netD = torch.nn.DataParallel(netD, self.gpu_ids)
            self.netD = networks.init_net(opt, netD, init_type=opt.init_type, init_gain=opt.init_gain,
                                          gpu_ids=self.gpu_ids, device=self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        bottleneck_width = 16
        current_level_res = bottleneck_width * 2 ** self.net_level  # 当前层的resolution

        real_A_image = self.normalizer.normalize(input['A' if self.AtoB else 'B'], 'seeg' if self.AtoB else 'eeg')
        real_B_image = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg')
        width_A = real_A_image.size()[2] if real_A_image.size()[2] % 2 == 0 else real_A_image.size()[2] - 1
        width_B = real_B_image.size()[2] if real_B_image.size()[2] % 2 == 0 else real_B_image.size()[2] - 1

        while width_A != bottleneck_width:  # eeg
            real_A_image = self.avg_layer(real_A_image)
            width_A /= 2
        while width_B != current_level_res:  # 压缩真实图像 seeg
            real_B_image = self.avg_layer(real_B_image)
            width_B /= 2

        self.real_A = real_A_image.to(self.device)
        self.real_B = real_B_image.to(self.device)
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    def set_net_level(self, net_level):
        self.net_level = net_level

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self):  # 先set_net_level再set_input再config_net再forward
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        """ generate random vector """
        if self.isTrain:
            batch_size = self.little_batch_size
        else:
            batch_size = self.opt.batch_size
        # fake_seed = torch.randn(batch_size, self.latent_size_, 1, 1, 1).to(self.device)  # 随机噪音
        self.fake_B = self.netG(self.real_A)  # 这里的输入改成脑电/噪音

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # WGANGP penalty and loss
        gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, self.real_B.detach(), self.fake_B.detach(), self.device)
        self.loss_D_grad = gradient_penalty
        self.loss_Wasserstein = - (self.loss_D_real + self.loss_D_fake)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + gradient_penalty
        #self.loss_D.backward(retain_graph=True)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        '''self.latent = self.netG.module.encoder(self.real_A.detach())  # 生成器encode出的中间变量
        self.pre_latent = self.netE_s(self.real_A)  # 预训练SEEG encoder输出的结果
        self.back_latent = self.netE_e(self.fake_B)'''
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):

        n_alt = 3
        self.forward()  # compute fake images: G(A)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        # update D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        if self.batch_idx % n_alt == 0:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights

    def config_net(self, batch_idx, fadein_steps, net_status, net_alpha, cur_step, data_size):

        self.little_batch_size = self.real_B.shape[0]  # little_batch_size存在是因为多显卡运算导致
        self.batch_idx = batch_idx

        if net_status == 'stable':
            net_alpha = 1.0
        elif net_status == 'fadein':
            if self.little_batch_size == self.opt.batch_size:
                net_alpha = 1.0 - (cur_step * data_size + batch_idx * self.little_batch_size) / (
                        fadein_steps * data_size)
            else:
                net_alpha = 1.0 - (cur_step * data_size + batch_idx * self.opt.batch_size + self.little_batch_size) /\
                            (fadein_steps * data_size)

        if net_alpha < 0.0:
            print("Alpha too small <0")
            return

        # change net status
        self.netG.module.net_config = [self.net_level, net_status, net_alpha]
        self.netD.module.net_config = [self.net_level, net_status, net_alpha]
        # print('net_level in config:', self.net_level)
        # if isinstance(self.netG, torch.nn.DataParallel):
        #     self.netG.module.net_config([self.net_level, net_status, net_alpha])
        #     print('netG config in module:', self.netG.module.net_config)
        # else:
        #     self.netG.net_config([self.net_level, net_status, net_alpha])
        #     print('netG config:', self.netG.net_config)
        # if isinstance(self.netD, torch.nn.DataParallel):
        #     self.netD.module.net_config([self.net_level, net_status, net_alpha])
        #
        #     print('netD config in module:', self.netD.module.net_config)
        # else:
        #     self.netD.net_config([self.net_level, net_status, net_alpha])
        #
        #     print('netD config:', self.netD.net_config)
