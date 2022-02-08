"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
import torch.nn as nn
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.numpy_tools import save_origin_npy, plot_spectrogram
from util.eeg_tools import Configuration
from gansynth.normalizer import DataNormalizer
from models import networks
# from train_asae import ASAE
import numpy as np
from data.eeg_dataset import EEGDatasetDataloader
import shutil
import functools
from models.networks import init_weights

class ASAE(nn.Module):

    def __init__(self, input_ch, output_ch, layer1_chan, layer2_chan):
        super(ASAE, self).__init__()

        norm_layer = functools.partial(nn.LayerNorm, elementwise_affine=False)

        aae = [nn.Linear(input_ch, layer1_chan, bias=False), norm_layer(layer1_chan), nn.Tanh(),
               nn.Linear(layer1_chan, output_ch, bias=False), norm_layer(output_ch)]
        self.aae = nn.Sequential(*aae)

        self.ae_ln1 = nn.Linear(output_ch, layer2_chan, bias=False)
        self.ae_ln2 = nn.Linear(layer2_chan, output_ch, bias=False)
        self.nonlinear = nn.Tanh()
        ae = [self.ae_ln1, norm_layer(layer2_chan), self.nonlinear,
              self.ae_ln2, norm_layer(output_ch), self.nonlinear]
        self.ae = nn.Sequential(*ae)

        init_weights(self, init_type='xavier')
        self.tiedWeights()

    def tiedWeights(self):

        self.ae_ln1.weight.data = self.ae_ln2.weight.data.t()

    def forward(self, x):
        x = self.nonlinear(self.aae(x))
        x = self.ae(x)
        return x


if __name__ == '__main__':

    conf = Configuration()
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    parser = TestOptions()  # get training options
    opt = parser.parse(save_opt=False)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    patients = []
    if opt.leave_out != '':
        patients.append(opt.leave_out)
    for patient in all_patients:
        if patient not in patients:
            patients.append(patient)

    for p_idx in range(1):  # patients中第一个病人一定是被leave out的
        opt.name = experiment_name_prefix + '_' + opt.leave_out
        opt.ae_name = ae_prefix + '_' + opt.leave_out
        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)
        if opt.phase == 'val':
            dataset = EEGDatasetDataloader(opt, patients[:p_idx] + patients[p_idx+1:])
            # dataset = EEGDatasetDataloader(opt, all_patients)
        else:
            dataset = EEGDatasetDataloader(opt, [opt.leave_out])  # create a dataset given opt.dataset_mode and other options

        load_filename = '%s_net_%s.pth' % (opt.epoch, 'asae')
        load_path = os.path.join(os.path.join('..', 'experiments', opt.checkpoints_dir, opt.name), load_filename)
        model = torch.nn.DataParallel(ASAE(conf.audio_length, conf.audio_length, 1500, 750).to(device), opt.gpu_ids).module    # create a model given opt.model and other options
        model.load_state_dict(torch.load(load_path, map_location=str(device)))
        normalizer = DataNormalizer(dataset, os.path.join('/home/hmq/Infos/norm_args/', dataset_name + '_without_' + opt.leave_out + '_temporal.npy'), False, domain=opt.domain)
        print("DataNormalizer prepared.")
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        npy_dir = os.path.join(web_dir, "npys")
        if os.path.exists(npy_dir):
            shutil.rmtree(npy_dir)
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            real_eeg = normalizer.normalize(data['B'][:, :conf.audio_length], 'eeg').to(device)
            img_path = data['B_paths']
            with torch.no_grad():
                fake_seeg = model(real_eeg)
            fake_seeg = normalizer.denormalize_temporal(fake_seeg, is_eeg=False)
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            save_origin_npy({'fake_seeg': fake_seeg}, npy_dir, img_path)
