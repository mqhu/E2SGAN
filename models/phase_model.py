import torch
from .base_model import BaseModel
from . import networks
import os
from gansynth.normalizer import DataNormalizer
from adabelief_pytorch import AdaBelief


class PhaseModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake', 'Wasserstein', 'D', 'D_grad', 'G_GAN', 'G_L1', 'G', 'style']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # , 'latent', 'back_latent']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.AtoB = self.opt.direction == 'AtoB'

        self.netG = networks.define_G(opt, 1, 1, opt.ngf, 'resnet', n_blocks=opt.n_blocks,
                                      norm=opt.norm,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                      device=self.device)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt, 2, opt.ndf, opt.netD, 2,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                          device=self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.optimizer_G = AdaBelief(self.netG.parameters(), lr=opt.g_lr, eps=1e-12, betas=(0.5, 0.999),
                                         # 这个参数影响很大！！
                                         weight_decay=True, rectify=False, fixed_decay=False, amsgrad=False)
            self.optimizer_D = AdaBelief(self.netD.parameters(), lr=opt.d_lr, eps=1e-12, betas=(0.5, 0.999),
                                         weight_decay=True, rectify=False, fixed_decay=False, amsgrad=False)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_D_vgg)
            self.train_G = False
            self.train_D = True
            self.use_L1 = True
            self.lambda_L1 = opt.lambda_L1

        self.normalizer = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        real_A = self.normalizer.normalize(input['A' if self.AtoB else 'B'], 'seeg' if self.AtoB else 'eeg').to(
            self.device)
        self.real_A = real_A[:, 1:, :, :]
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        real_B = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg').to(
            self.device)
        self.real_B = real_B[:, 1:, :, :]
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def update_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def start_training_G(self):
        self.train_G = True

    def flip_training_D(self):
        self.train_D = not self.train_D

    def flip_training_G(self):
        self.train_G = not self.train_G

    def update_lambda_L1(self, value):
        if value == 0:
            self.use_L1 = False
            print("Stop using L1 loss")
        else:
            self.lambda_L1 = value
            print("Update lambda L1 to {}...".format(self.lambda_L1))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # if self.batch_idx % 100:
        #     print('fake B:', self.fake_B.shape)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B !!!!! attach importance!!!!
        # fake.detach() sets fake_grad_fn to none, which enables it to be sent as input as a pure tensor
        # and avoids duplicate grad calculations
        # 注:real_A和B维度是n_batch,n_chan,height,width，所以上面的拼接是按channel拼，所以要求图size完全一样
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # real A先按频率去中心化试试
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # WGANGP penalty and loss
        gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB.detach(), self.device)
        self.loss_D_grad = gradient_penalty
        self.loss_Wasserstein = - (self.loss_D_real + self.loss_D_fake)
        # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_real + self.loss_D_fake * 0.75 + self.loss_D_eeg * 0.25) * 0.5 + gradient_penalty
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + gradient_penalty
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        '''self.latent = self.netG.module.encoder(self.real_A.detach())  # 生成器encode出的中间变量
        self.pre_latent = self.netE_s(self.real_A)  # 预训练SEEG encoder输出的结果
        self.back_latent = self.netE_e(self.fake_B)'''

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        #  计算style loss
        # real_mean, real_std = self.cal_mean_std(self.real_B)
        # fake_mean, fake_std = self.cal_mean_std(self.fake_B)
        # self.loss_style = 0.5 * (self.criterionL2(real_mean, fake_mean) + self.criterionL2(real_std, fake_std)) * self.opt.lambda_L1
        # Second, G(A) = B
        # combine loss and calculate gradients
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 #+ self.loss_style
        self.loss_G.backward()

    def cal_mean_std(self, feat, eps=1e-8):
        size = feat.size()
        N, C, W = size[: 3]
        feat_var = feat.var(dim=3) + eps
        feat_std = feat_var.sqrt().view(N, C, W, 1)
        feat_mean = feat.mean(dim=3).view(N, C, W, 1)
        return feat_mean, feat_std

    def optimize_parameters(self):

        # self.set_VGG_requires_grad(self.netD_vgg, True)  # enable backprop for D_vgg
        n_alter = 1
        self.forward()  # compute fake images: G(A)

        if self.train_D:
            n_alter = 5
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            # update D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        # update G
        if self.train_G and self.batch_idx % n_alter == 0:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
