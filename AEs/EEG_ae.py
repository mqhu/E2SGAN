import torch.nn as nn
from models.networks import ResnetBlock
from torchvision import models
import torch
import torch.nn as nn
import functools

'''
class EEG_Encoder(nn.Module):

    def __init__(self, input_nc=2, ngf=32, n_downsampling=3, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(EEG_Encoder, self).__init__()

        model = [nn.ReplicationPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [# nn.MaxPool2d(2),  试试看不用Pooling
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        n_blocks = 3
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias, convD=2)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


class EEG_Decoder(nn.Module):

    def __init__(self, eeg_chans, input_nc=2, ngf=32, n_downsampling=3, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(EEG_Decoder, self).__init__()

        model = []
        # self.eeg_chans = eeg_chans
        # self.chan_count = self.get_upsample_chan(eeg_chans, n_downsampling)

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
                      #nn.Conv3d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      #norm_layer(int(ngf * mult / 2)),
                      #nn.ReLU(True)]

        self.model = nn.Sequential(*model)

        # oneD_opr = [nn.Conv3d(self.chan_count, eeg_chans, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #              norm_layer(eeg_chans),
        #              nn.ReLU(True)]
        #
        # self.oneD_opr = nn.Sequential(*oneD_opr)

        final = []
        final += [nn.ReplicationPad2d(3)]
        final += [nn.Conv2d(ngf, input_nc, kernel_size=7, padding=0)]
        final += [nn.Tanh()]  # Tanh的值域是(-1,1)，所以生成信号的值域也是这个，因此输入信号要scale到(-1,1)

        self.final = nn.Sequential(*final)

    def forward(self, x):

        x = self.model(x)
        x = self.final(x)
        # if self.chan_count != self.eeg_chans:
        #     x = self.oneD_opr(x.transpose(2, 1))
        #     x = self.final(x.transpose(1, 2))
        # else:
        #     x = self.final(x)
        return x

    def get_upsample_chan(self, input_chan, n_scaling, scale_factor=2):

        result = input_chan
        for i in range(n_scaling):
            result = result // scale_factor  # 向下取整
        return result * (scale_factor ** n_scaling)'''


class EEG_AE(nn.Module):

    def __init__(self, pre_path, input_nc=2, ngf=32, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(EEG_AE, self).__init__()
        # n_downsampling = 3
        # encoder = [EEG_Encoder(input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling, norm_layer=norm_layer,
        #                        use_bias=use_bias)]
        # decoder = [EEG_Decoder(eeg_chans, input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling,
        #                        norm_layer=norm_layer, use_bias=use_bias)]

        net = models.vgg16_bn()
        pre = torch.load(pre_path)
        net.load_state_dict(pre)

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        en_features = [nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0)]  # 增加1x1卷积升维
        en_features += list(net.features)[0: 33]  # 取到filter数为512为止

        k_size = 512
        n_itr = 3
        de_features = []
        for i in range(n_itr):
            de_features += [nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(k_size, k_size // 2, kernel_size=3, padding=1),
                            norm_layer(k_size // 2), nn.ReLU(True),
                            nn.Conv2d(k_size // 2, k_size // 2, kernel_size=3, padding=1),
                            norm_layer(k_size // 2), nn.ReLU(True)]
            k_size = k_size // 2

        de_features += [nn.Conv2d(3, input_nc, kernel_size=1, stride=1, padding=0), nn.Tanh()]

        self.encoder = nn.Sequential(*en_features)
        self.decoder = nn.Sequential(*de_features)

    def forward(self, x):
        """Standard forward"""
        #print('before:', x.shape)
        x = self.encoder(x)
        #print('mid:', x.shape)
        x = self.decoder(x)
        #print('after:', x.shape)
        return x

    def init_network(self):

        m = self.encoder[0]  # 初始化第一层卷积参数
        print(self.encoder[0])
        print(self.encoder[1])
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

        for m in self.decoder:
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


if __name__ == '__main__':
    net = EEG_AE("/home/cbd109-3/Users/data/hmq/codes/vgg/vgg16_bn-6c64b313.pth")
    net.init_network()
    net.print_parameters()
