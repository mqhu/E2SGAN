import time
from options.train_options import TrainOptions
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Statistics
from gansynth.normalizer import DataNormalizer
import os
import torch
from data.eeg_dataset import EEGDatasetDataloader
from util.distance_metrics import calculate_distance
import nni
import shutil


def autoupdate_params(opt, param_to_update, tuner_params):
    '''
    挑取自动调超参的参数
    :param opt:
    :param param_ls:
    :return:
    '''
    params = {}
    opt_dict = vars(opt)
    for p in param_to_update:
        params[p] = opt_dict[p]

    assert len(param_to_update) == len(params.keys())

    params.update(tuner_params)
    for k, v in params.items():
        print('Updating opt.{} from {}'.format(k, getattr(opt, k)))
        setattr(opt, k, v)
        print('to {}'.format(getattr(opt, k)))

    return opt


if __name__ == '__main__':
    parser = TrainOptions()  # get training options
    opt = parser.parse(save_opt=False)
    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    score_names = ['temporal', 'mag 0-4', 'mag 5-7', 'mag 8-15', 'mag 16-28', 'mag 0-28',
                   'psd 0-4', 'psd 5-7', 'psd 8-15', 'psd 16-28', 'psd 0-28',
                   'phase_mean', 'rmse_mean']
    params_to_autotune = ['g_lr', 'd_lr', 'lambda_L1', 'netD']

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    rank = int(os.environ["RANK"])
    nproc_per_node = 1
    normalizer_dir = '/home/hmq/Infos/norm_args/'

    for p_idx in range(len(all_patients[:1])):
        opt.name = experiment_name_prefix + '_' + all_patients[p_idx]
        opt.ae_name = ae_prefix + '_' + all_patients[p_idx]
        opt.leave_out = all_patients[p_idx]
        opt.phase = 'val'
        valdataset = EEGDatasetDataloader(opt, all_patients[:p_idx] + all_patients[p_idx + 1:])
        opt.phase = 'train'
        dataset = EEGDatasetDataloader(opt, all_patients[:p_idx] + all_patients[p_idx+1:])  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        opt.save_latest_freq = dataset_size

        #  自动调参更新参数
        '''
        tuner_params = nni.get_next_parameter()
        if tuner_params is not None:
            opt = autoupdate_params(opt, params_to_autotune, tuner_params)
        '''
        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)

        print('The number of training images = %d' % dataset_size)
        normalizer_name = dataset_name + '_without_' + all_patients[p_idx]
        if opt.is_IF:
            normalizer_name += '_IF'
        normalizer_name += '.npy'
        if opt.leave_out == 'lk':
            normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), False, domain=opt.domain)
        else:
            normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), True, domain=opt.domain)
        print("DataNormalizer prepared.")

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.set_normalizer(normalizer)
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters_single = 0                # the total number of training iterations
        total_iters_all = 0
        statistics = Statistics(opt)   # 保存画出来的loss
        #score_names = metric.get_score_names()
        # score_names = ('Train Precision', 'Train Recall', 'Test Precision', 'Test Recall')
        score_list = []
        lambda_L1 = opt.lambda_L1
        min_scores = dict.fromkeys(score_names[1:], float('inf'))
        min_scores['default'] = 0
        # update_lambda_L1_freq = (opt.n_epochs + opt.n_epochs_decay) // 6
        # model.start_training_G()
        correct_epoch_ls = []
        correct_val_ls = []

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter_single = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            epoch_iter_all = 0
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            dataset.train_sampler.set_epoch(epoch)
            correct_epoch = 0
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
                visuals = model.get_current_visuals()
                correct_epoch += visuals['correct']
                if rank == 0 and total_iters_all % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    losses['train accuracy'] = correct_epoch / (epoch_iter_all / nproc_per_node) * 100
                    print('correct:', visuals['correct'])
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter_all, losses, t_comp, t_data, rank)

                if total_iters_all % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    if rank == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters_all))
                        save_suffix = 'iter_%d' % total_iters_all if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)
                    torch.distributed.barrier()

                if rank == 0 and total_iters_all % opt.loss_freq == 0:
                    losses = model.get_current_losses()
                    losses['train accuracy'] = correct_epoch / (epoch_iter_all / nproc_per_node) * 100
                    statistics.save_plotted_losses(epoch, int(epoch_iter_all / opt.print_freq),
                                                   float(epoch_iter_all) / dataset_size, losses)

                iter_data_time = time.time()

            if rank == 0 and epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters_all))
                model.save_networks('latest')
                model.save_networks(epoch)

            if rank == 0 and (epoch % opt.val_epoch_freq == 0):
                correct_val = 0
                for i, data in enumerate(valdataset):
                    model.set_input(data)
                    model.test()
                    visuals = model.get_current_visuals()
                    correct_val += visuals['correct']
                correct_val /= len(valdataset)
                correct_val_ls.append([correct_val * 100])
                print('correct_val:', correct_val)
                statistics.save_score_plots(opt.val_epoch_freq, {'names': ['val accuracy'], 'scores': correct_val_ls})
            torch.distributed.barrier()

            # if epoch == 10:
            #     lambda_L1 /= 2
            #     model.update_lambda_L1(lambda_L1)
            # elif epoch == 20:
            #     model.update_lambda_L1(0)

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
            # nni.report_final_result(min_scores)
            if rank == 0:
                print('Rank %d \t End of epoch %d / %d \t Time Taken: %d sec' % (rank, epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.
