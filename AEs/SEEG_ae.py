import torch.nn as nn
from models.networks import ResnetBlock


def get_downsample_chan(input_chan, n_scaling, scale_factor=2):
    result = input_chan
    for i in range(n_scaling):
        result = result // scale_factor
    return result


def get_upsample_chan(input_chan, n_scaling, scale_factor=2):

    result = input_chan
    for i in range(n_scaling):
        result = result // scale_factor  # 向下取整
    return result * (scale_factor ** n_scaling)


class SEEG_Encoder(nn.Module):

    def __init__(self, seeg_chans, eeg_chans, input_nc=2, ngf=32, n_downsampling=3, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(SEEG_Encoder, self).__init__()

        model = [nn.ReplicationPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [# nn.MaxPool2d(2),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        n_blocks = 3
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias, convD=2)]

        # n_lat_eeg = get_downsample_chan(eeg_chans, n_downsampling)
        # n_lat_seeg = get_downsample_chan(seeg_chans, n_downsampling)
        #
        # oneD_opr = [nn.Conv2d(n_lat_seeg, n_lat_eeg, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #             norm_layer(n_lat_eeg),
        #             nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        #self.oneD_opr = nn.Sequential(*oneD_opr)

    def forward(self, x):

        x = self.model(x)
        #x = self.oneD_opr(x.transpose(2, 1))
        return x


class SEEG_Decoder(nn.Module):

    def __init__(self, seeg_chans, eeg_chans, input_nc=2, ngf=32, n_downsampling=3, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(SEEG_Decoder, self).__init__()

        model = []
        # self.seeg_chans = seeg_chans
        # self.chan_count = get_upsample_chan(seeg_chans, n_downsampling)
        # n_lat_eeg = get_downsample_chan(eeg_chans, n_downsampling)
        # n_lat_seeg = get_downsample_chan(seeg_chans, n_downsampling)

        # oneD_opr1 = [nn.Conv3d(n_lat_eeg, n_lat_seeg, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #             norm_layer(n_lat_seeg),
        #             nn.ReLU(True)]
        #
        # self.oneD_opr1 = nn.Sequential(*oneD_opr1)

        n_blocks = 3
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias, convD=2)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='nearest'),  # 反卷积改为上采样+卷积
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*model)

        # oneD_opr2 = [nn.Conv3d(self.chan_count, seeg_chans, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #              norm_layer(seeg_chans),
        #              nn.ReLU(True)]
        #
        # self.oneD_opr2 = nn.Sequential(*oneD_opr2)

        final = []
        final += [nn.ReplicationPad2d(3)]
        final += [nn.Conv2d(ngf, input_nc, kernel_size=7, padding=0)]
        final += [nn.Tanh()]  # Tanh的值域是(-1,1)，所以生成信号的值域也是这个，因此输入信号要scale到(-1,1)

        self.final = nn.Sequential(*final)

    def forward(self, x):
        """Standard forward"""
        # x = self.oneD_opr1(x.transpose(2, 1))
        # x = self.model(x.transpose(1, 2))
        # if self.chan_count != self.seeg_chans:
        #     x = self.oneD_opr2(x.transpose(2, 1))
        #     x = self.final(x.transpose(1, 2))
        # else:
        #     x = self.final(x)
        x = self.model(x)
        x = self.final(x)
        return x


class SEEG_AE(nn.Module):

    def __init__(self, eeg_chans, seeg_chans, input_nc=2, ngf=32, norm_layer=nn.InstanceNorm3d, use_bias=True):

        super(SEEG_AE, self).__init__()
        n_downsampling = 3

        encoder = [SEEG_Encoder(seeg_chans, eeg_chans, input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling,
                                norm_layer=norm_layer, use_bias=use_bias)]
        decoder = [SEEG_Decoder(seeg_chans, eeg_chans, input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling,
                                norm_layer=norm_layer, use_bias=use_bias)]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """Standard forward"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
