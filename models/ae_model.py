import torch
import torch.nn as nn
from .base_model import BaseModel
from AEs.EEG_ae import EEG_AE
from AEs.SEEG_ae import SEEG_AE
from models import networks
# from modules import SelfAttention
from util.eeg_tools import Configuration

class AEModel(BaseModel):

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

        return parser

    def __init__(self, opt, to_gpu=True, ngf=32, norm_layer=nn.InstanceNorm2d, pretrain=False):

        BaseModel.__init__(self, opt)

        self.loss_names = ['ae_e', 'E', 'ae_s', 'S']
        self.model_names = ['AE_e', 'AE_s']
        self.visual_names = ['real_eeg', 'real_seeg', 'fake_eeg', 'fake_seeg']

        conf = Configuration()

        if pretrain:
            self.netAE_e = networks.VGG16_AE(opt.vgg_version, "/home/hmq/Projects/pretrained_vgg/vgg16-397923af.pth", input_nc=opt.input_nc)
            self.netAE_s = networks.VGG16_AE(opt.vgg_version, "/home/hmq/Projects/pretrained_vgg/vgg16-397923af.pth", input_nc=opt.output_nc)
        else:
            self.netAE_e = networks.VGG16_AE(opt.vgg_version, input_nc=opt.input_nc, output_nc=opt.input_nc)
            self.netAE_s = networks.VGG16_AE(opt.vgg_version, input_nc=opt.output_nc, output_nc=opt.output_nc)

        # self.netAttentional = Attention(conf.w, conf.h)

        if to_gpu:
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.netAE_e.to(self.gpu_ids[0])
                self.netAE_e = torch.nn.DataParallel(self.netAE_e, self.gpu_ids)
                self.netAE_s.to(self.gpu_ids[0])
                self.netAE_s = torch.nn.DataParallel(self.netAE_s, self.gpu_ids)
                self.netAttentional.to(self.gpu_ids[0])
                self.netAttentional = torch.nn.DataParallel(self.netAttentional, self.gpu_ids)

            if opt.isTrain:
                self.criterionL2 = nn.MSELoss()
                self.optimizer_E = torch.optim.Adam(self.netAE_e.parameters(), lr=opt.e_lr, betas=(opt.beta1, 0.999))
                self.optimizer_S = torch.optim.Adam(self.netAE_s.parameters(), lr=opt.s_lr, betas=(opt.beta1, 0.999))
                self.optimizers = [self.optimizer_E, self.optimizer_S]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # if AtoB:
        self.real_seeg = self.normalizer.normalize(input['A'], 'seeg').to(self.device)
        self.real_eeg = self.normalizer.normalize(input['B'], 'eeg').to(self.device)
        # self.eeg = torch.sum(eeg, dim=2)
        # else:
        #     self.seeg = self.normalizer.normalize(input['B'], 'seeg').to(self.device)
        #     self.eeg = self.normalizer.normalize(input['A'], 'eeg').to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths = input['A_paths']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self):
        # gen_eeg = self.netAttentional(self.eeg, self.eeg, self.eeg)
        self.fake_eeg = self.netAE_e(self.real_eeg)
        self.fake_seeg = self.netAE_s(self.real_seeg)
        #self.lat_eeg = self.netAE_e.module.encoder(self.eeg)
        #self.lat_seeg = self.netAE_s.module.encoder(self.seeg)

    def backward_E(self):
        self.loss_ae_e = self.criterionL2(self.fake_eeg, self.real_eeg) * 10
        #self.loss_lat_e = self.criterionL2(self.lat_eeg, self.lat_seeg.detach())
        self.loss_E = self.loss_ae_e# + self.loss_lat_e
        self.loss_E.backward()

    def backward_S(self):
        self.loss_ae_s = self.criterionL2(self.fake_seeg, self.real_seeg) * 10
        #self.loss_lat_s = self.criterionL2(self.lat_seeg, self.lat_eeg.detach())
        self.loss_S = self.loss_ae_s# + self.loss_lat_s
        self.loss_S.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_E.zero_grad()
        self.backward_E()
        self.optimizer_E.step()

        self.optimizer_S.zero_grad()
        self.backward_S()
        self.optimizer_S.step()
