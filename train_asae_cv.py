import time
import os
from options.train_options import TrainOptions
# from data import create_dataset
from gansynth.normalizer import DataNormalizer
from models.asae_model import ASAEModel
import numpy as np
import torch
from util.visualizer import Statistics, Visualizer
from data.eeg_dataset import EEGDatasetDataloader
from util.distance_metrics import calculate_distance


if __name__ == '__main__':
    parser = TrainOptions()   # get training options
    opt = parser.parse(save_opt=False)
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    experiment_name_prefix = opt.name
    score_names = ['temporal', 'mag 0-4', 'mag 5-7', 'mag 8-15', 'mag 16-28', 'mag 0-28',
                   'psd 0-4', 'psd 5-7', 'psd 8-15', 'psd 16-28', 'psd 0-28',
                   'phase_mean', 'rmse_mean', 'bd_mean', 'hd_mean']
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    rank = int(os.environ["RANK"])
    nproc_per_node = 2
    normalizer_dir = '/home/hmq/Infos/norm_args/'

    for p_idx in range(1):
        opt.name = experiment_name_prefix + '_' + all_patients[p_idx]
        opt.leave_out = all_patients[p_idx]
        dataset = EEGDatasetDataloader(opt, all_patients[:p_idx] + all_patients[p_idx+1:])  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        opt.save_latest_freq = dataset_size - (dataset_size % opt.batch_size)
        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)
        visualizer = Visualizer(opt)
        statistics = Statistics(opt)
        print('The number of training images = %d' % dataset_size)

        normalizer_name = dataset_name + '_without_' + all_patients[p_idx]
        if opt.is_IF:
            normalizer_name += '_IF'
        normalizer_name += '.npy'
        if opt.leave_out == 'lk':
            normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), False,
                                        domain=opt.domain)
        else:
            normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), True, domain=opt.domain)
        print("DataNormalizer prepared.")

        model = ASAEModel(opt, pretrain=False)
        model.setup(opt)
        model.set_normalizer(normalizer)
        total_iters_single = 0                # the total number of training iterations
        total_iters_all = 0
        ae_e_loss = []
        ae_s_loss = []
        lat_e_loss = []
        lat_s_loss = []
        e_loss = []
        s_loss = []
        score_list = []

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter_single = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            epoch_iter_all = 0
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
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if rank == 0 and epoch_iter_all % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    # losses = (
                    # float(model.loss_ae_e), float(model.loss_ae_s), float(model.loss_lat_e), float(model.loss_lat_s),
                    # float(model.loss_ae_e + model.loss_lat_e), float(model.loss_ae_s + model.loss_lat_s))
                    # ae_e_loss.append(losses[0])
                    # ae_s_loss.append(losses[1])
                    # lat_e_loss.append(losses[2])
                    # lat_s_loss.append(losses[3])
                    # e_loss.append(losses[4])
                    # s_loss.append(losses[5])
                    # print("total %d iterations done" % total_iters)
                    # print("(ae_e_loss: %f, ae_s_loss: %f, lat_e_loss: %f, lat_s_loss: %f, e_loss: %f, s_loss: %f)" % losses)
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter_all, losses, t_comp, t_data, rank)
                    # if opt.display_id > 0:
                    #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters_all % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
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

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                if rank == 0:
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters_all))
                    model.save_networks('latest')
                    model.save_networks(epoch)
                torch.distributed.barrier()

            if epoch % opt.val_epoch_freq == 0:
                if rank == 0:
                    cmd = "python test_cv.py --dataroot " + opt.dataroot + " --direction BtoA --dataset_mode eeg"\
                          + " --norm " + opt.norm\
                          + " --input_nc " + str(opt.input_nc) + " --output_nc " + str(opt.output_nc)\
                          + " --preprocess none --no_flip --gpu_ids 0,1 --phase val"\
                          + " --model " + opt.model + " --name " + experiment_name_prefix \
                          + ' --n_blocks ' + str(opt.n_blocks) + ' --ngf ' + str(opt.ngf) \
                          + ' --leave_out ' + opt.leave_out
                    if opt.is_IF:
                        cmd += ' --is_IF'
                    if opt.pghi:
                        cmd += ' --pghi'
                    os.system(cmd)
                    results = calculate_distance(os.path.join(opt.dataroot, 'A', 'val'),
                                                 os.path.join('/home/hmq/Projects/experiments/results/',
                                                              opt.name, 'val_latest', 'npys'), normalizer, method='ae',
                                                 is_IF=opt.is_IF, aggregate=True)
                    score_list.append([results['temporal_mean']] + list(results['mag_mean']) + list(results['psd_mean']) +
                                      [results['phase_mean']] + [results['rmse_mean']] + [results['bd_mean']] + [
                                          results['hd_mean']])
                    statistics.save_score_plots(opt.val_epoch_freq, {'names': score_names, 'scores': score_list})
                torch.distributed.barrier()

            # if epoch % opt.test_freq == 0:
            #     print('testing current model')
            #     os.system('sh test_ae.sh test')
            #     oldDir = os.path.join('results', opt.name, 'test', 'npys')
            #     newDir = os.path.join('results', opt.name, 'test', 'npys' + str(epoch))
            #     os.rename(oldDir, newDir)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.

        # dir_path = os.path.join("losses", opt.name)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # np.save(os.path.join(dir_path, "ae_e_loss"), ae_e_loss)
        # np.save(os.path.join(dir_path, "ae_s_loss"), ae_s_loss)
        # np.save(os.path.join(dir_path, "lat_e_loss"), lat_e_loss)
        # np.save(os.path.join(dir_path, "lat_s_loss"), lat_s_loss)
        # np.save(os.path.join(dir_path, "e_loss"), e_loss)
        # np.save(os.path.join(dir_path, "s_loss"), s_loss)
