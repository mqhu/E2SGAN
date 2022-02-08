import torch
from .base_model import BaseModel
from . import networks
from adabelief_pytorch import AdaBelief
import os.path as op


class ClassifierModel(BaseModel):
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
        self.loss_names = ['CrossEntropy']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Classifier']
        self.visual_names = ['correct']
        # define networks (both generator and discriminator)
        self.AtoB = self.opt.direction == 'AtoB'
        self.patientLabel = {'lk': 0, 'zxl': 1, 'yjh': 2, 'tll': 3, 'lmk': 4, 'lxh': 5, 'wzw': 6}

        self.netClassifier = networks.define_D(opt, opt.input_nc, opt.ndf, 'classsifier', 2,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                      device=self.device)

        if self.isTrain:
            # define loss functions
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.optimizer = AdaBelief(self.netClassifier.parameters(), lr=opt.d_lr, eps=1e-12, betas=(0.5, 0.999),
                                         weight_decay=True, rectify=False, fixed_decay=False, amsgrad=False)
            self.optimizers.append(self.optimizer)

        self.normalizer = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        real_B = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg').to(
            self.device)
        self.real_B = real_B[:, :1, :, :]
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']
        labels = []
        for path in self.image_paths:
            fileName = op.basename(path)
            patient = fileName.split('_')[0]
            labels.append(self.patientLabel[patient])
        self.target = torch.LongTensor(labels).to(self.device)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.predict = self.netClassifier(self.real_B)
        self.correct = torch.eq(self.predict.argmax(dim=1), self.target).sum().item()

    def backward(self):

        self.loss_CrossEntropy = self.criterionCE(self.predict, self.target)
        self.loss_CrossEntropy.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
