import torch
import torch.nn as nn
from .base_model import BaseModel
from AEs.EEG_ae import EEG_AE
from AEs.SEEG_ae import SEEG_AE
from models import networks
# from adabelief_pytorch import AdaBelief


class ASAEModel(BaseModel):

    def __init__(self, opt, input_nc=2, ngf=32, norm_layer=nn.InstanceNorm2d, pretrain=False):

        BaseModel.__init__(self, opt)

        self.loss_names = ['MSE']
        self.model_names = ['ASAE']
        self.visual_names = ['real_eeg', 'real_seeg', 'fake_seeg']
        self.AtoB = self.opt.direction == 'AtoB'

        # if pretrain:
        #     netASAE = networks.VGG16_AE(opt.vgg_version, "/home/hmq/Projects/pretrained_vgg/vgg16-397923af.pth", input_nc=input_nc)
        # else:
        #     netASAE = networks.VGG16_AE(opt.vgg_version, input_nc=input_nc)

        # if len(self.gpu_ids) > 0:
        #     assert (torch.cuda.is_available())
        #     netASAE.to(self.gpu_ids[0])
        #     self.netASAE = torch.nn.DataParallel(netASAE, self.gpu_ids)
        self.netASAE = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, 'resnet', n_blocks=opt.n_blocks, norm=opt.norm,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, device=self.device)

        if opt.isTrain:
            self.criterionMSE = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.netASAE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))# AdaBelief(self.netASAE.parameters(), lr=opt.lr, eps=1e-12, betas=(0.9, 0.999))
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # if AtoB:
        self.real_seeg = self.normalizer.normalize(input['A'], 'seeg').to(self.device)
        # self.real_seeg = self.real_seeg[:, 1:, :, :]
        self.real_eeg = self.normalizer.normalize(input['B'], 'eeg').to(self.device)
        # self.real_eeg = self.real_eeg[:, 1:, :, :]
        # else:
        #     self.seeg = self.normalizer.normalize(input['B'], 'seeg').to(self.device)
        #     self.eeg = self.normalizer.normalize(input['A'], 'eeg').to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self):
        self.fake_seeg = self.netASAE(self.real_eeg)

    def backward(self):
        # self.loss_mag = self.criterionMSE(self.real_seeg[:, :1, :, :], self.fake_seeg[:, :1, :, :]) * 10
        # self.loss_phase = self.criterionMSE(self.real_seeg[:, 1:, :, :], self.fake_seeg[:, 1:, :, :]) / 10
        self.loss_MSE = self.criterionMSE(self.real_seeg, self.fake_seeg) * 10
        self.loss_MSE.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
