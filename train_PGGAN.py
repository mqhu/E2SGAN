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
from GANMetrics import metric
from data.eeg_dataset import EEGDatasetDataloader
import torch
import os
from models import networks
from util.distance_metrics import calculate_distance


def train(opt, model, dataset, fadein_steps, net_level, net_status, net_alpha, cur_step, total_iters_single, total_iters_all, visualizer, statistics, epoch):

    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter_single = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    epoch_iter_all = 0
    visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.set_net_level(net_level)
    dataset.train_sampler.set_epoch(epoch)
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if epoch_iter_all > opt.max_dataset_size:
            break
        if total_iters_all % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters_single += opt.batch_size
        epoch_iter_single += opt.batch_size
        total_iters_all += opt.batch_size * nproc_per_node
        epoch_iter_all += opt.batch_size * nproc_per_node
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.config_net(i, fadein_steps, net_status, net_alpha, cur_step, len(dataset))
        model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        if rank == 0 and epoch_iter_all % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter_all, losses, t_comp, t_data, rank)

        if total_iters_all % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            if rank == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters_all))
                save_suffix = 'iter_%d' % total_iters_all if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            torch.distributed.barrier()

        if rank == 0 and total_iters_all % opt.loss_freq == 0:
            losses = model.get_current_losses()
            statistics.save_plotted_losses(epoch, int(epoch_iter_all / opt.print_freq),
                                           float(epoch_iter_all) / dataset_size, losses)

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        if rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters_all))
            model.save_networks('latest')
            model.save_networks(epoch)
        torch.distributed.barrier()

    if (epoch > 38) and (epoch % opt.val_epoch_freq == 0):
        if rank == 0:
            cmd = "python test_PGGAN.py --dataroot " + opt.dataroot + " --direction BtoA --dataset_mode eeg --norm instance" \
                  + " --input_nc " + str(opt.input_nc) + " --output_nc " + str(opt.output_nc) \
                  + " --preprocess none --no_flip --gpu_ids 0,1 --phase val" \
                  + " --model " + opt.model + " --ae_name " + ae_prefix + " --name " + experiment_name_prefix \
                  + ' --n_blocks ' + str(opt.n_blocks) \
                  + ' --ndf ' + str(opt.ndf) + ' --ngf ' + str(opt.ngf) \
                  + ' --leave_out ' + opt.leave_out
            if opt.is_IF:
                cmd += ' --is_IF'
            if opt.pghi:
                cmd += ' --pghi'
            os.system(cmd)
            results = calculate_distance(os.path.join(opt.dataroot, 'A', 'val'),
                                         os.path.join('/public/home/xlwang/hmq/Projects/experiments/results/',
                                                      opt.name, 'val_latest', 'npys'), normalizer, method='GAN',
                                         aggregate=True, is_IF=opt.is_IF)
            score_list.append([results['temporal_mean']] + list(results['mag_mean']) + list(results['psd_mean']) +
                              [results['phase_mean']] + [results['rmse_mean']] + [results['bd_mean']] + [
                                  results['hd_mean']])
            statistics.save_score_plots(opt.val_epoch_freq, {'names': score_names, 'scores': score_list})
        torch.distributed.barrier()

    print('End of epoch %d / %d \t Time Taken: %d sec' % (
    epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()  # update learning rates at the end of every epoch.


if __name__ == '__main__':
    stable_and_fadein_step = [2, 4, 6, 6]
    # stable_and_fadein_step = [1, 1, 1, 1]
    net_status = "stable"
    net_alpha = 1.0  # 似乎没有什么用

    parser = TrainOptions()  # get training options
    opt = parser.parse(save_opt=False)
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    score_names = ['temporal', 'mag 0-4', 'mag 5-7', 'mag 8-15', 'mag 16-28', 'mag 0-28',
                   'psd 0-4', 'psd 5-7', 'psd 8-15', 'psd 16-28', 'psd 0-28',
                   'phase_mean', 'rmse_mean', 'bd_mean', 'hd_mean']
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    rank = int(os.environ["RANK"])
    nproc_per_node = 2
    normalizer_dir = '/public/home/xlwang/hmq/Infos/norm_args/'
    # normalizer_dir = '/home/hmq/Infos/norm_args/'

    for p_idx in range(1):
        if opt.leave_out == 'none':
            opt.leave_out = all_patients[p_idx]
        else:
            p_idx = all_patients.index(opt.leave_out)
        opt.name = experiment_name_prefix + '_' + opt.leave_out
        opt.ae_name = ae_prefix + '_' + opt.leave_out
        dataset = EEGDatasetDataloader(opt, all_patients[:p_idx]+all_patients[p_idx+1:])  # create a dataset given opt.dataset_mode and other options
        # dataset = EEGDatasetDataloader(opt, all_patients)
        dataset_size = len(dataset)    # get the number of images in the dataset.
        opt.save_latest_freq = dataset_size
        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)

        print('The number of training images = %d' % dataset_size)
        normalizer_name = dataset_name + '_without_' + opt.leave_out
        if opt.is_IF:
            normalizer_name += '_IF'
        if opt.pghi:
            normalizer_name += '_pghi'
        normalizer_name += '.npy'
        compute_normalizer = False if os.path.exists(os.path.join(normalizer_dir, normalizer_name)) else True
        normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), compute_normalizer,
                                    domain=opt.domain, use_phase=not opt.pghi)
        print("DataNormalizer prepared.")

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        model.set_normalizer(normalizer)
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        total_iters_single = 0  # the total number of training iterations
        total_iters_all = 0
        statistics = Statistics(opt)  # 保存画出来的loss
        # score_names = metric.get_score_names()
        # score_names = ('Train Precision', 'Train Recall', 'Test Precision', 'Test Recall')
        score_list = []
        lambda_L1 = opt.lambda_L1
        min_scores = dict.fromkeys(score_names[1:], float('inf'))
        min_scores['default'] = 0
        # update_lambda_L1_freq = (opt.n_epochs + opt.n_epochs_decay) // 6
        # model.start_training_G()
        networks.setup_seed(6)
        epoch = 0

        for cur_level in range(len(stable_and_fadein_step)):
            stable_steps = stable_and_fadein_step[cur_level]
            fadein_steps = stable_and_fadein_step[cur_level]

            if cur_level == len(stable_and_fadein_step) - 1:
                #dataloader.change_batch_size(batch_size=4)
                stable_steps = 20


            if cur_level == 0:
                for step in range(stable_steps):
                    epoch_start_time = time.time()  # timer for entire epoch
                    epoch += 1
                    train(opt, model, dataset, fadein_steps, cur_level, "stable", net_alpha, step, total_iters_single, total_iters_all, visualizer, statistics, epoch)
            else:
                net_status = "fadein"  # ???这行有啥用
                for step in range(fadein_steps):
                    epoch_start_time = time.time()  # timer for entire epoch
                    epoch += 1
                    train(opt, model, dataset, fadein_steps, cur_level, "fadein", net_alpha, step, total_iters_single, total_iters_all, visualizer, statistics, epoch)
                for step in range(stable_steps * 2):
                    epoch_start_time = time.time()  # timer for entire epoch
                    net_alpha = 1.0
                    epoch += 1
                    train(opt, model, dataset, fadein_steps, cur_level, "stable", net_alpha, step, total_iters_single, total_iters_all, visualizer, statistics, epoch)
