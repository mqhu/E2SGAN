import time
import os
from options.train_options import TrainOptions
# from data import create_dataset
from gansynth.normalizer import DataNormalizer
from models.ae_model import AEModel
import numpy as np
import torch
from util.visualizer import Statistics, Visualizer
from data.eeg_dataset import EEGDatasetDataloader


if __name__ == '__main__':
    parser = TrainOptions()   # get training options
    opt = parser.parse(save_opt=False)
    patients = ['lk', 'tll', 'zxl', 'yjh']
    experiment_name_prefix = opt.name

    for p_idx in range(1):
        opt.name = experiment_name_prefix + '_' + patients[p_idx]
        parser.save_options(opt)
        dataset = EEGDatasetDataloader(opt, patients[:p_idx]+patients[p_idx+1:], 3600)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        visualizer = Visualizer(opt)
        statistics = Statistics(opt)
        print('The number of training images = %d' % dataset_size)

        normalizer = DataNormalizer(dataset, os.path.join('/home/hmq/Infos/norm_args/', 'without_' + patients[p_idx] + '.npy'), True)
        print("DataNormalizer prepared.")

        model = AEModel(opt, pretrain=False)
        model.setup(opt)
        model.set_normalizer(normalizer)
        total_iters = 0                # the total number of training iterations
        ae_e_loss = []
        ae_s_loss = []
        lat_e_loss = []
        lat_s_loss = []
        e_loss = []
        s_loss = []

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
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
                    # t_comp = (time.time() - iter_start_time) / opt.batch_size
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
