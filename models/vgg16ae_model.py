import torch
import torch.nn as nn
from .base_model import BaseModel
from torchvision import models as tm
from .networks import get_norm_layer
from modules import SelfAttention
from util.eeg_tools import Configuration

conf = Configuration()


class VGG16_AE(BaseModel):

    def __init__(self, opt, pre_path=None, use_bias=True):

        BaseModel.__init__(self, opt)
        # n_downsampling = 3
        # encoder = [EEG_Encoder(input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling, norm_layer=norm_layer,
        #                        use_bias=use_bias)]
        # decoder = [EEG_Decoder(eeg_chans, input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling,
        #                        norm_layer=norm_layer, use_bias=use_bias)]
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'Wasserstein', 'D', 'G', 'D_grad']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']  # , 'latent', 'back_latent']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            # self.model_names = ['G', 'D', 'D_vgg']
        else:  # during test time, only load G
            self.model_names = ['G']

        net = tm.vgg16()  # 避免加入batch norm
        sfattn = SelfAttention(conf.w, conf.h)
        if pre_path is not None:
            pre = torch.load(pre_path)
            net.load_state_dict(pre)

        norm_layer = get_norm_layer('instance')
        en_features = [nn.Conv2d(opt.input_nc, 3, kernel_size=1, stride=1, padding=0, bias=False)]  # 增加1x1卷积升维
        en_features += list(net.features)[: 23]  # 取到filter数为512为止

        k_size = 512
        n_itr = 3
        de_features = []
        for i in range(n_itr):
            de_features += [#nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(k_size, k_size // 2,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                            norm_layer(k_size // 2), nn.LeakyReLU(0.2, True),
                            nn.Conv2d(k_size // 2, k_size // 2, kernel_size=3, padding=1, bias=False),
                            norm_layer(k_size // 2), nn.LeakyReLU(0.2, True)]
            k_size = k_size // 2

        de_features += [nn.Conv2d(64, opt.output_nc, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh()]

        self.encoder = nn.Sequential(*en_features)
        self.decoder = nn.Sequential(*de_features)

        self.init_network(pre_path)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        eeg = self.normalizer.normalize(input['B'], 'eeg').to(self.device)
        self.eeg = torch.sum(eeg, dim=2)
        self.image_paths = input['A_paths']  # 以SEEG文件名为准

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, x):
        """Standard forward"""
        #print('before:', x.shape)
        x = self.encoder(x)
        #print('mid:', x.shape)
        x = self.decoder(x)
        #print('after:', x.shape)
        return x

    def init_network(self, prepath=None):

        m = self.encoder[0]  # 初始化第一层卷积参数
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

        init_blocks = []
        if prepath is None:
            init_blocks += self.encoder[1:]
        init_blocks += self.decoder

        for m in init_blocks:
            classname = getattr(getattr(m, '__class__'), '__name__')
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        # for p in list(self.parameters())[2: 11]:  # 这里数字要check一下，注意bias和batch norm也有参数
        #     p.requires_grad = False
        '''在训练模型时记得放到gpu上'''
    def print_parameters(self):
        print(self.encoder)
        print(self.decoder)
        for p in self.parameters():
            print(p.size())