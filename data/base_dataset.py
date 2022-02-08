"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import cv2
import sys
import os
import librosa
from gansynth import phase_operation
from util.eeg_tools import Configuration

conf = Configuration()


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size): #用来返回crop的位置？
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=cv2.INTER_CUBIC, convert=True, t2f=True, iseeg=True, rel_pos=None, **kwargs): #对图片进行resize或等比例resize或crop
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __trim_data(img)))  # 对齐数据

    if t2f:
        if opt.pghi:
            transform_list.append(transforms.Lambda(lambda img: __pghi_preprocess(img, kwargs['preprocessor'])))
        else:
            transform_list.append(transforms.Lambda(lambda img: __to_mag_and_IF(img, iseeg=iseeg, is_IF=opt.is_IF, rel_pos=rel_pos)))
    else:
        transform_list.append(transforms.Lambda(lambda img: __squeeze_dim(img)))

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess: #形变成正方形
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Lambda(lambda img: __square_size(img, opt.load_size, method)))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    #if opt.preprocess == 'none':
    #    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert: #默认转化为Tensor
        #transform_list += [transforms.ToTensor()]
        transform_list.append(transforms.Lambda(lambda img: __np_to_tensor(img)))
        """ 图片需要归一化，脑电我觉得不要
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        """
    #transform_list.append(transforms.Lambda(lambda img: __increase_dimension(img)))

    return transforms.Compose(transform_list)


"""这里的crop和resize都要改成适用于脑电的"""
def __make_power_2(img, base, method=cv2.INTER_CUBIC): #保证图片尺寸是base的倍数
    ow, oh = img.size()
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize(w, h, method)


def __scale_width(img, target_width, method=cv2.INTER_CUBIC): #等比例resize
    ow, oh = img.size()
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize(w, h, method)


def __square_size(img, target_width,method=cv2.INTER_CUBIC): #正方形剪裁
    return img.resize(target_width, target_width, method)


def __crop(img, pos, size): #剪裁
    print(os.path.basename(os.path.abspath(__file__)).split(".")[0] + "  " + str(sys._getframe().f_lineno))
    ow, oh = img.size()
    x1, y1 = pos
    tw = th = size
    print("original size: " + str(img.size()))
    print("crop points: " + str((x1, y1, x1 + tw, y1 + th)))
    if (ow > tw or oh > th):
        return img.crop(x1, y1, x1 + tw, y1 + th)
    return img


def __flip(img, flip): #翻转
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT) #左右翻转
    return img


def __np_to_tensor(img):
    t = torch.from_numpy(img)
    t = t.type(torch.FloatTensor)
    return t


def __eegsegment_to_tensor(img):
    t = torch.from_numpy(img.get_data())
    t = t.type(torch.FloatTensor)
    return t


def __increase_dimension(img):

    if torch.is_tensor(img):
        return img.unsqueeze(0)
    elif type(img) is np.ndarray:
        return np.expand_dims(img, axis=0)


def __1ch_to_3ch(img):
    '''img is numpy.ndarray'''
    img = img.get_data()
    if len(img.shape) == 2:
        img = __increase_dimension(img)
    if img.shape[0] == 3:
        return img
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def __to_mag_and_IF(img, iseeg=True, is_IF=False, rel_pos=None):
    """
    获取mag和IF特征并合并在一起
    注意输入格式为(n_chan, n_timepoint)
    :return ndarray
    """
    mag_IF = []
    '''记得和eeg_tools里的mag_plus_phase统一！！！'''
    if iseeg:
        n_fft = conf.eeg_n_fft
        #win_length = 512
        hop = conf.eeg_hop
    else:
        n_fft = conf.seeg_n_fft
        #win_length = 512
        hop = conf.seeg_hop

    max_len = 128
    img = img.get_data()

    for i, chan in enumerate(img):
        # if conf.audio_length < len(chan):
        #     chan = chan[: conf.audio_length]
        spec = librosa.stft(chan, n_fft=n_fft, win_length=n_fft, hop_length=hop)
        # magnitude = np.log(np.abs(spec))[: max_len]
        magnitude = np.log(np.abs(spec) + conf.epsilon)[: max_len]  # abs是求复数的模，log我觉得是因为原始值太小了
        if is_IF:
            angle = np.angle(spec)
            IF = phase_operation.instantaneous_frequency(angle, time_axis=1)[: max_len]
        else:
            angle = np.angle(spec)[: max_len]
            IF = angle

        # if iseeg:
        #     stack_real_image = np.stack((magnitude, IF, rel_dist(magnitude.shape, rel_pos[i])), axis=0)
        # else:
        #     stack_real_image = np.stack((magnitude, IF), axis=0)
        stack_real_image = np.stack((magnitude, IF), axis=0)

        mag_IF.append(stack_real_image)

    # if is_IF:
    #     mag_IF = np.asarray(mag_IF).transpose((1, 0, 2, 3))  # (2, n_chan, n_time, n_freq)
    # else:
    #     mag_IF = mag_IF[0]
    # if iseeg:
    #     mag_IF = np.asarray(mag_IF)
    # else:
    #     mag_IF = mag_IF[0]
    mag_IF = mag_IF[0]

    return mag_IF


def rel_dist(mag_shape, rel_pos):
    '''只针对输入的EEG'''
    r_d = np.linalg.norm(rel_pos, ord=1)
    r_d_mat = np.zeros(mag_shape) + r_d
    return r_d_mat


def __squeeze_dim(img):
    '''
    对于原始时域数据形如(1, 1784)，压缩掉度数为1的维度，ASAE要用到
    :param img:
    :return:
    '''
    img = img.get_data()
    return img.squeeze()


def __pghi_preprocess(img, preprocessor):
    """
    获取mag和IF特征并合并在一起
    注意输入格式为(n_chan, n_timepoint)
    :return ndarray
    """
    img = img.get_data()
    preprocessed = preprocessor(img[0])[:, :conf.h, :conf.w]

    return preprocessed

def __trim_data(img):

    data = img.get_data()
    assert len(data.shape) == 2

    if data.shape[1] > conf.audio_length:
        data = data[:, :conf.audio_length]
    img.set_data(data)
    return img
