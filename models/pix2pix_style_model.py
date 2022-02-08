import torch
from .base_model import BaseModel
from . import networks
import os
from gansynth.normalizer import DataNormalizer
from adabelief_pytorch import AdaBelief


class Pix2PixStyleModel(BaseModel):
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
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        # parser.add_argument('--loss_freq', type=int, default=50, help='frequency of saving loss plots')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'Wasserstein', 'D', 'G', 'D_vgg', 'D_vgg_real',
        #                   'D_vgg_fake', 'D_vgg_Wasserstein', 'D_grad', 'D_vgg_grad', 'G_vgg']
        self.loss_names = ['D_real', 'D_fake', 'Wasserstein', 'D', 'D_grad', 'D_eeg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # , 'latent', 'back_latent']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            # self.model_names = ['G', 'D', 'D_vgg']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.AtoB = self.opt.direction == 'AtoB'

        '''
        # 下面是用AEnet
        from .ae_model import AEModel
        AE_net = AEModel(opt, to_gpu=True)
        if self.isTrain:
            if not opt.train_AE:
                AE_net.load_networks('latest', path=os.path.join('../experiments', opt.checkpoints_dir, opt.ae_name))
        if self.AtoB:
            self.netEn_forward = AE_net.netAE_s.module.encoder  # seeg Encoder
            # netEn_backward = AE_net.netAE_e.module.encoder  # eeg Encoder
            self.netDe = AE_net.netAE_e.module.decoder  # eeg Decoder
        else:
            self.netEn_forward = AE_net.netAE_e.module.encoder  # eeg Encoder
            # netEn_backward = AE_net.netAE_s.module.encoder  # seeg Encoder
            self.netDe = AE_net.netAE_s.module.decoder  # seeg Decoder
        '''
        '''
        # 下面是用ASAENet
        from .asae_model import ASAEModel
        ASAE_net = ASAEModel(opt, input_nc=opt.input_nc, pretrain=False)
        if self.isTrain:
            if not opt.train_AE:
                ASAE_net.load_networks('latest', path=os.path.join('../experiments', opt.checkpoints_dir, opt.ae_name))
        self.netEn_forward = ASAE_net.netASAE.module.encoder  # eeg Encoder
        self.netDe = ASAE_net.netASAE.module.decoder  # seeg Decoder

        netG = networks.VGG_generator(opt.vgg_version, self.netEn_forward, self.netDe)

        # netE_s = networks.AE_component(opt, 'seeg', 'encoder', 1, 1)  # SEEG encoder
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            netG.to(self.gpu_ids[0])
            # netEn_backward.to(self.gpu_ids[0])
            #netE_s.to(self.gpu_ids[0])
            self.netG = torch.nn.DataParallel(netG, self.gpu_ids)
            # self.netEn_backward = torch.nn.DataParallel(netEn_backward, self.gpu_ids)
            #self.netE_s = torch.nn.DataParallel(netE_s, self.gpu_ids)
        # self.set_requires_grad([self.netEn_backward], False)
        '''
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet', n_blocks=3, norm=opt.norm,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, 2,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_vgg = networks.define_D_vgg(opt.input_nc, self.gpu_ids, pretrained=True,
            #        pre_path="/home/cbd109/Users/hmq/codes/vgg16/vgg16-397923af.pth",  norm=opt.norm, use_dropout=opt.no_dropout)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = AdaBelief(self.netG.parameters(), lr=opt.g_lr, eps=1e-16, betas=(0.9, 0.999),
                                         # 这个参数影响很大！！
                                         weight_decay=True, rectify=False, fixed_decay=False, amsgrad=False)
            self.optimizer_D = AdaBelief(self.netD.parameters(), lr=opt.d_lr, eps=1e-16, betas=(0.9, 0.999),
                                         weight_decay=True, rectify=False, fixed_decay=False, amsgrad=False)
            # self.optimizer_D_vgg = torch.optim.Adam(self.netD_vgg.parameters(), lr=0.00004, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_D_vgg)
            self.train_G = False
            self.lambda_L1 = opt.lambda_L1

        self.normalizer = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A = self.normalizer.normalize(input['A' if self.AtoB else 'B'], 'seeg' if self.AtoB else 'eeg').to(
            self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg').to(
            self.device)
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def update_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def start_training_G(self):
        self.train_G = True
        self.loss_names += ['G_GAN', 'G_L1', 'G']

    def update_lambda_L1(self, value):
        self.lambda_L1 = value
        print("Update lambda L1 to {}...".format(self.lambda_L1))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # self.latent = self.netG.module.encoder(self.real_A)  # 生成器encode出的中间变量
        # self.pre_latent = self.netE_s(self.real_A.detach())  # 预训练SEEG encoder输出的结果
        # self.back_latent = self.netEn_backward(self.fake_B)  # 最后可以一个loss一个loss的backward回去看看是哪里出问题

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B !!!!! attach importance!!!!
        # fake.detach() sets fake_grad_fn to none, which enables it to be sent as input as a pure tensor
        # and avoids duplicate grad calculations
        # 注:real_A和B维度是n_batch,n_chan,height,width，所以上面的拼接是按channel拼，所以要求图size完全一样
        real_A = (self.real_A - torch.mean(self.real_A)) / torch.std(self.real_A)
        real_B = (self.real_B - torch.mean(self.real_B)) / torch.std(self.real_B)
        fake_B = (self.fake_B - torch.mean(self.fake_B)) / torch.std(self.fake_B)
        fake_AB = torch.cat((real_A, fake_B), 1)
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # 区分seeg和eeg
        # real_AA = torch.cat((self.real_A, self.real_A), 1)
        real_AA = torch.cat((real_A, real_A), 1)
        pred_eeg = self.netD(real_AA)
        self.loss_D_eeg = self.criterionGAN(pred_eeg, False)  # 看看vaeGAN
        # WGANGP penalty and loss
        gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB.detach(), self.device)
        self.loss_D_grad = gradient_penalty
        self.loss_Wasserstein = - (self.loss_D_real + self.loss_D_fake)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_real + (self.loss_D_fake + self.loss_D_eeg) * 0.5) * 0.5 + gradient_penalty
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        '''self.latent = self.netG.module.encoder(self.real_A.detach())  # 生成器encode出的中间变量
        self.pre_latent = self.netE_s(self.real_A)  # 预训练SEEG encoder输出的结果
        self.back_latent = self.netE_e(self.fake_B)'''

        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_AB = torch.cat(((self.real_A - torch.mean(self.real_A) / torch.std(self.real_A)),
                             (self.fake_B - torch.mean(self.fake_B) / torch.std(self.fake_B))), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # pred_fake_B = self.netD_vgg(self.fake_B)
        # self.loss_G_vgg = self.criterionGAN(pred_fake_B, True)
        # fake_mean = torch.mean(self.fake_B)
        # fake_std = torch.std(self.fake_B)
        # real_mean = torch.mean(self.real_B)
        # real_std = torch.std(self.real_B)
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1((self.fake_B - fake_mean) / fake_std, (self.real_B - real_mean) / real_std) * self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # encoder的中间向量要和预训练模型的结果相近
        # self.loss_G_lat = self.criterionMSE(self.latent, self.pre_latent.detach())
        # generator输出再经过预训练EEG encoder的结果要与本身的中间向量一致
        # self.loss_G_lat_consis = self.criterionMSE(self.latent, self.back_latent) * self.opt.lambda_lat
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  # + self.loss_G_lat_consis
        self.loss_G.backward()

    def optimize_parameters(self):

        # self.set_VGG_requires_grad(self.netD_vgg, True)  # enable backprop for D_vgg
        n_alter = 5
        self.forward()  # compute fake images: G(A)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        # update D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        # for i, param in enumerate(self.netD.parameters()):
        #     print('D idx:{}\ngrad:{}'.format(i, param.grad))
        # print('D loss grad:', self.loss_D.grad)
        self.optimizer_D.step()  # update D's weights
        # update G
        if self.train_G and self.batch_idx % n_alter == 0:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            # for i, param in enumerate(self.netG.parameters()):
            #     print('G idx:{}\ngrad:{}'.format(i, param.grad))
            # print('G loss grad:', self.loss_G.grad)
            self.optimizer_G.step()  # udpate G's weights
