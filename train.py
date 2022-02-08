"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Statistics
from gansynth.normalizer import DataNormalizer
import sys
import os
import torch
from GANMetrics import metric
from util import eeg_tools
import numpy as np
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler


# default_collate_func = dataloader.default_collate
#
# def default_collate_override(batch):
#   dataloader._use_shared_memory = False
#   return default_collate_func(batch)
#
# setattr(dataloader, 'default_collate', default_collate_override)
#
# for t in torch._storage_classes:
#     if sys.version_info[0] == 2:
#         if t in ForkingPickler.dispatch:
#             del ForkingPickler.dispatch[t]
#     else:
#         if t in ForkingPickler._extra_reducers:
#             del ForkingPickler._extra_reducers[t]


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    normalizer = DataNormalizer(dataset, opt.normalizer_path, False)
    print("DataNormalizer prepared.")

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.set_normalizer(normalizer)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    statistics = Statistics(opt)   # 保存画出来的loss
    #score_names = metric.get_score_names()
    score_names = ('Train Precision', 'Train Recall', 'Test Precision', 'Test Recall')
    score_list = []
    start_G_flag = True

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        if start_G_flag and epoch >= opt.epoch_ahead:
            model.start_training_G()
            start_G_flag = False
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.update_batch_idx(i)
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            """
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            """

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            if total_iters % opt.loss_freq == 0:
                losses = model.get_current_losses()
                statistics.save_plotted_losses(epoch, int(epoch_iter / opt.print_freq),
                                               float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            # os.system("sh test_stft_64x64.sh train")
            # train_precision, train_recall = eeg_tools.calculate_spindle(
            #     "/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/B/train/",
            #     "/home/cbd109/Users/hmq/codes/pix2pix/results/stft_Fz_64x64_res_ae/train_latest/npys/",
            #     "/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/ref_B6/",
            #     normalizer)
            # os.system("sh test_stft_64x64.sh test")
            # test_precision, test_recall = eeg_tools.calculate_spindle(
            #     "/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/B/test/",
            #     "/home/cbd109/Users/hmq/codes/pix2pix/results/stft_Fz_64x64_res_ae/test_latest/npys/",
            #     "/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/ref_B6/",
            #     normalizer)
            # score_list.append([train_precision, train_recall, test_precision, test_recall])
            # statistics.save_score_plots(opt.save_epoch_freq,
            #                             {'names': score_names,
            #                              'scores': score_list})
        '''if epoch % opt.print_score_freq == 0:
            print("saving score plots at the end of eopch %d" % epoch)
            os.system("sh test_IF.sh")
            scores = metric.compute_score_raw(1, "/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/test/",\
                                              "/home/cbd109/Users/hmq/codes/pix2pix/results/IF_GAN/test_latest/npys/",\
                                              None, normalizer=normalizer, conv_model='resnet34', workers=4)
            score_list.append(scores)
            score_dict = {'names': score_names, 'scores': score_list}
            statistics.save_score_plots(opt.print_score_freq, score_dict)'''

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
