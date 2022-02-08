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
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.numpy_tools import save_origin_npy, plot_spectrogram
from util.eeg_tools import save_origin_mat
from gansynth.normalizer import DataNormalizer
import matplotlib.pyplot as plt
from models import networks
import numpy as np
from data.eeg_dataset import EEGDatasetDataloader
import shutil


if __name__ == '__main__':
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    parser = TestOptions()  # get training options
    opt = parser.parse(save_opt=False)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    patients = []
    normalizer_dir = '/public/home/xlwang/hmq/Infos/norm_args/'
    if opt.leave_out != 'none':
        patients.append(opt.leave_out)
    for patient in all_patients:
        if patient not in patients:
            patients.append(patient)

    p_idx = 0
    opt.name = experiment_name_prefix + '_' + opt.leave_out
    opt.ae_name = ae_prefix + '_' + opt.leave_out
    parser.save_options(opt)
    dataset_name = os.path.basename(opt.dataroot)
    if opt.phase == 'val':
        dataset = EEGDatasetDataloader(opt, patients[:p_idx] + patients[p_idx + 1:])
        # dataset = EEGDatasetDataloader(opt, all_patients)
    else:
        dataset = EEGDatasetDataloader(opt,
                                       [opt.leave_out])  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    normalizer_name = dataset_name + '_without_' + opt.leave_out
    if opt.is_IF:
        normalizer_name += '_IF'
    if opt.pghi:
        normalizer_name += '_pghi'
    normalizer_name += '.npy'
    normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), False, domain=opt.domain,
                                use_phase=not opt.pghi)
    print("DataNormalizer prepared.")
    model.set_normalizer(normalizer)
    #model.set_normalizer(normalizer)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    npy_dir = os.path.join(web_dir, "npys")
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.set_net_level(2)
    model.netG.net_config = [2, 'stable', 1]
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results 模仿一下这一段把原始的numpy数据给保存下来
        img_path = model.get_image_paths()     # get image paths
        # img_path = [str(i)]
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        save_origin_npy(visuals, npy_dir, img_path)
        #save_resized_npy(visuals, npy_dir, img_path, (22, 256))
    webpage.save()  # save the HTML
