import time
import os
import torch
from options.train_options import TrainOptions
from gansynth.normalizer import DataNormalizer
from util.visualizer import Statistics, Visualizer
import torch.nn as nn
from models.networks import init_weights
# from adabelief_pytorch import AdaBelief
from data.eeg_dataset import EEGDatasetDataloader
from util.distance_metrics import calculate_distance
from torch.optim import lr_scheduler
import functools
from util.eeg_tools import Configuration


all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
conf = Configuration()

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

        init_weights(self, init_type='normal')
        self.tiedWeights()

    def tiedWeights(self):

        self.ae_ln1.weight.data = self.ae_ln2.weight.data.t()

    def forward(self, x):
        x = self.nonlinear(self.aae(x))
        x = self.ae(x)
        return x


def train(parser, opt, rank):

    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    score_names = ['temporal', 'mag 0-4', 'mag 5-7', 'mag 8-15', 'mag 16-28', 'mag 0-28',
                   'psd 0-4', 'psd 5-7', 'psd 8-15', 'psd 16-28', 'psd 0-28',
                   'phase_mean', 'rmse_mean', 'cos_mean', 'phase_dtw_mean', 'phase_rmse_mean']
    nproc_per_node = len(opt.gpu_ids)
    device = torch.device('cuda:{}'.format(opt.local_rank))

    # criterion_1 = nn.CrossEntropyLoss().to(device)
    criterion_2 = nn.MSELoss().to(device)
    normalizer_dir = '/home/hmq/Infos/norm_args/'

    for p_idx in range(2, len(all_patients)):

        min_scores = dict.fromkeys(score_names[1:], float('inf'))
        min_scores['default'] = 0
        asae = torch.nn.parallel.DistributedDataParallel(ASAE(conf.audio_length, conf.audio_length, 1500, 750).to(device), device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank)
        optimizer_1 = torch.optim.Adam(asae.module.aae.parameters(), lr=opt.lr / 2, betas=(0.9, 0.999))
            # AdaBelief(asae.module.aae.parameters(), lr=opt.lr / 2., eps=1e-8, betas=(0.9, 0.999),
            #                              weight_decay=1e-2, weight_decouple=True, rectify=False, fixed_decay=False, amsgrad=False)
        optimizer_2 = torch.optim.Adam(asae.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        # AdaBelief(asae.parameters(), lr=opt.lr, eps=1e-8, betas=(0.9, 0.999),
        #                              weight_decay=1e-2, weight_decouple=True, rectify=False, fixed_decay=False, amsgrad=False)
        scheduler_1 = lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=opt.n_epochs, eta_min=0)
        scheduler_2 = lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=opt.n_epochs, eta_min=0)

        opt.name = experiment_name_prefix + '_' + all_patients[p_idx]
        # opt.ae_name = ae_prefix + '_' + all_patients[p_idx]
        opt.leave_out = all_patients[p_idx]
        dataset = EEGDatasetDataloader(opt, all_patients[:p_idx]+all_patients[p_idx+1:])  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        opt.save_latest_freq = dataset_size
        parser.save_options(opt)
        dataset_name = os.path.basename(opt.dataroot)
        print('The number of training images = %d' % dataset_size)
        normalizer_name = dataset_name + '_without_' + opt.leave_out + '_temporal'
        normalizer_name += '.npy'
        compute_normalizer = False if os.path.exists(os.path.join(normalizer_dir, normalizer_name)) else True
        normalizer = DataNormalizer(dataset, os.path.join(normalizer_dir, normalizer_name), compute_normalizer,
                                    domain=opt.domain, use_phase=not opt.pghi)
        # aux_normalizer = DataNormalizer(None, os.path.join(normalizer_dir, dataset_name + '_without_' + opt.leave_out + '.npy'), False,
        #                                 use_phase=not opt.pghi)
        print("DataNormalizer prepared.")

        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters_single = 0                # the total number of training iterations
        total_iters_all = 0
        statistics = Statistics(opt)   # 保存画出来的loss
        score_list = []
        min_scores = dict.fromkeys(score_names[1:], float('inf'))
        min_scores['default'] = 0
        model_save_dir = os.path.join('..', 'experiments', opt.checkpoints_dir, opt.name)
        loss_1 = 0
        loss_2 = 0

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter_single = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            epoch_iter_all = 0
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            dataset.train_sampler.set_epoch(epoch)

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if epoch_iter_all > opt.max_dataset_size:
                    break
                if total_iters_all % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                real_seeg = normalizer.normalize(data['A'][:, :conf.audio_length], 'seeg').to(device)
                real_eeg = normalizer.normalize(data['B'][:, :conf.audio_length], 'eeg').to(device)
                if i % 2 != 0:
                    fake_seeg_aae = torch.nn.functional.softmax(asae.module.aae(real_eeg), dim=1)
                    real_seeg = torch.nn.functional.softmax(real_seeg, dim=1)
                    loss_1 = -torch.sum(real_seeg * torch.log(fake_seeg_aae) + (1 - real_seeg) * torch.log(1 - fake_seeg_aae)) / opt.batch_size
                    optimizer_1.zero_grad()
                    loss_1.backward()
                    optimizer_1.step()
                else:
                    fake_seeg = asae(real_eeg)
                    loss_2 = criterion_2(fake_seeg, real_seeg) * 10.
                    optimizer_2.zero_grad()
                    loss_2.backward()
                    optimizer_2.step()
                    asae.module.tiedWeights()

                total_iters_single += opt.batch_size
                epoch_iter_single += opt.batch_size
                total_iters_all += opt.batch_size * nproc_per_node
                epoch_iter_all += opt.batch_size * nproc_per_node

                if rank == 0 and total_iters_all % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = {'loss_CE': loss_1, 'loss_MSE': loss_2}
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter_all, losses, t_comp, t_data, rank)

                if total_iters_all % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    if rank == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters_all))
                        save_suffix = 'iter_%d' % total_iters_all if opt.save_by_iter else 'latest'
                        save_filename = '%s_net_%s.pth' % (save_suffix, 'asae')
                        save_path = os.path.join(model_save_dir, save_filename)
                        torch.save(asae.module.state_dict(), save_path)
                    torch.distributed.barrier()

                if rank == 0 and total_iters_all % opt.loss_freq == 0:
                    losses = {'loss_CE': loss_1, 'loss_MSE': loss_2}
                    statistics.save_plotted_losses(epoch, int(epoch_iter_all / opt.print_freq),
                                                   float(epoch_iter_all) / dataset_size, losses)

                iter_data_time = time.time()

            if rank == 0 and epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters_all))
                save_filename = '%s_net_%s.pth' % (epoch, 'asae')
                save_path = os.path.join(model_save_dir, save_filename)
                torch.save(asae.module.state_dict(), save_path)

            if rank == 0 and (epoch >= opt.epoch_ahead) and (epoch % opt.val_epoch_freq == 0):
                os.system("python test_asae.py --dataroot " + opt.dataroot + " --direction BtoA --dataset_mode eeg --norm instance --input_nc 2 --output_nc 2 --preprocess none --no_flip --gpu_ids 0 --phase val "
                         + "--model " + opt.model + " --ae_name " + ae_prefix + " --name " + experiment_name_prefix +
                          ' --leave_out ' + opt.leave_out + ' --domain temporal')
                results = calculate_distance(os.path.join(opt.dataroot, 'A', 'val'), os.path.join('/home/hmq/Projects/experiments/results/',
                                                        opt.name, 'val_latest', 'npys'), normalizer, method='asae', aggregate=True, aux_normalizer=None)
                score_list.append([results['temporal_mean']] + list(results['mag_mean']) + list(results['psd_mean']) +
                                  [results['phase_mean'], results['rmse_mean'], results['cos_mean'], results['phase_dtw_mean'],
                                   results['phase_rmse_mean']])
                statistics.save_score_plots(opt.val_epoch_freq, {'names': score_names, 'scores': score_list})
            torch.distributed.barrier()

            if rank == 0:
                print('Rank %d \t End of epoch %d / %d \t Time Taken: %d sec' % (rank, epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            scheduler_1.step()
            scheduler_2.step()


if __name__ == '__main__':
    parser = TrainOptions()  # get training options
    opt = parser.parse(save_opt=False)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    rank = int(os.environ["RANK"])
    train(parser, opt, rank)
    # opt.isTrain = False
    # opt.domain = 'freq'
    # dataset = EEGDatasetDataloader(opt, all_patients[:0] + all_patients[0 + 1:])
    # for i, data in enumerate(dataset):
    #     print(data)
    #     print(data['A'].shape)
    #     print('i:', i)
    #     if i > 5:
    #         break
    # cmd = 'python test_asae.py --dataroot /home/hmq/Datasets/cv_0704_60 --direction BtoA --dataset_mode eeg --norm instance --input_nc 2 --output_nc 2 --preprocess none --no_flip --name cv_asae_0707 --gpu_ids 0 --phase $1  --leave_out lk --domain temporal'
    # os.system(cmd)
