import numpy as np

from util.eeg_tools import *
from util.numpy_tools import *
from util.util import *
from util.visualizer import plot_tsne
import shutil
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import mne
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import librosa
from gansynth.normalizer import DataNormalizer
#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis
from GANMetrics import metric
import data.base_dataset as bd
import scipy.io
import mat_file
import math
from util.distance_metrics import *
from data.base_dataset import __to_mag_and_IF
from gansynth import phase_operation
import torch.nn.functional as F


"""
目前scale_width + crop可以跑
纯crop会因为图A和图B原始尺寸不同，且寻找切割点时是按照A的尺寸去随机找的，导致B可能会是空的tensor而报错
"""

src_path = r"/home/cbd109/Users/hmq/huashan_data_processed/LK/LK_rest/"
dest_path = r"/home/cbd109/Users/hmq/huashan_data_processed/LK/LK_rest/"
filename = "LK_1_seizure_raw.fif"
conf = Configuration()


def generate_temporal_comparable_plot(path, prefix):
    # fake itvl=0.0005, realA itvl=0.00005, realB itvl=0.0002
    # path = r"/home/cbd109/Users/hmq/codes/pix2pix/results/vgg_1/test_latest/npys/"
    eeg_file = "LK_Sleep_Aug_4th_2am_eeg_raw-0.fif"
    seeg_file = "LK_Sleep_Aug_4th_2am_seeg_raw-6-0.fif"

    real_A = np.load(os.path.join(path, prefix + '_real_A.npy'))
    real_B = np.load(os.path.join(path, prefix + '_real_B.npy'))
    fake_B = np.load(os.path.join(path, prefix + '_fake_B.npy'))

    draw_comparable_figure(real_A=real_A, real_B=real_B, fake_B=fake_B)
    #draw_comparable_figure(fake_B=fake_B, ch_intv=0.0002)


def generate_freq_comparable_plot(path, prefix, save_dir=None):
    real_A = np.load(os.path.join(path, prefix + '_real_A.npy'))
    real_B = np.load(os.path.join(path, prefix + '_real_B.npy'))
    fake_B = np.load(os.path.join(path, prefix + '_fake_B.npy'))

    plot_spectrogram(np.arange(0, 224), np.arange(0, 28, .125), prefix + ' Spectrogram', vmin=-.9, vmax=.9, save_dir=save_dir,
                     RealEEG=real_A.mean(axis=0), RealSEEG=real_B[0], FakeSEEG=fake_B[0])


def slice_eeg_seeg(pairs, raw_dir, eeg_save_dir, seeg_save_dir, patient):

    # pairs = [('EEG Fp2-Ref', 'POL E10'), ('EEG FZ-Ref', 'EEG F1-Ref'), ('EEG F4-Ref-1', 'POL F11'), ('EEG F8-Ref-1', 'POL B14'), ('EEG T4-Ref', 'POL H14')]
    all = []
    for p in pairs:
        all += list(p)
    raw = read_raw_signal(raw_dir)  # Sleep_Aug_4th_2am
    raw.pick_channels(all)
    for pair in pairs:
        eeg_name = pair[0]
        seeg_name = pair[1]
        raw_eeg = raw.copy().pick_channels([pair[0]])
        raw_seeg = raw.copy().pick_channels([pair[1]])
        raw_eeg.resample(conf.eeg_sf, npad="auto")
        raw_eeg.filter(1., None, fir_design='firwin')
        # rand_starts = random.sample(range(raw_eeg.get_data().shape[1] - 1784), 300)
        # slice_random_data(raw_eeg, eeg_save_dir, rand_starts, 1784, prefix=patient + '_' + eeg_name + '_' + seeg_name + '_')
        slice_data(raw_eeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=0, prefix=patient + '_' + eeg_name + '_' + seeg_name + '_')
        raw_seeg.resample(conf.seeg_sf, npad="auto")
        raw_seeg.filter(.5, None, fir_design='firwin')
        slice_data(raw_seeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=0, prefix=patient + '_' + eeg_name + '_' + seeg_name + '_')
        # slice_random_data(raw_seeg, seeg_save_dir, rand_starts, 1784, prefix=patient + '_' + eeg_name + '_' + seeg_name + '_')
        del raw_eeg
        del raw_seeg


def eeg_baselined_results():
    EEG_baseline = np.load(os.path.join(evaluationDir, 'cv_0704_60_EEG_avg_test_dist_v2.npy'), allow_pickle=True).item()
    evaluations = {'asae_eval': 'cv_0704_60_asae_avg_test_dist_v1.npy',
                    'ae_eval': 'cv_0704_60_ae_avg_test_dist_v1.npy', 'eeggan_eval': 'cv_0704_60_eeggan_avg_test_dist_v1.npy',
                    'gansynth_eval': 'cv_0704_60_gansynth_1_avg_test_dist_v1.npy', 'gan_phase': 'cv_0704_60_pix2pix_phase_avg_test_dist_v2.npy',
                   'gan_glb_attn': 'cv_0704_60_pix2pix_global_onesided_avg_test_dist_v2.npy',
                   'gan_IF': 'cv_0704_60_pix2pix_IF_avg_test_dist_v1.npy',
                   'gan_glb': 'cv_0704_60_pix2pix_global_avg_test_dist_v2.npy'}
    for k, v in evaluations.items():
        evaluations[k] = np.load(os.path.join(evaluationDir, v), allow_pickle=True).item()

    for k, v in EEG_baseline.items():
        metric = k.split('_')[0]
        if 'mean' in k:
            mean_baseline = v

            for kk in evaluations.keys():
                if type(mean_baseline) == list:
                    for i in range(len(mean_baseline)):
                        try:
                            evaluations[kk][k][i] /= mean_baseline[i]
                            evaluations[kk][metric + '_std'][i] /= mean_baseline[i]
                        except:
                            print('{} mean in {} error'.format(metric, kk))

                else:
                    try:
                        evaluations[kk][k] /= mean_baseline
                        evaluations[kk][metric + '_std'] /= mean_baseline
                    except:
                        print('{} std in {} error'.format(metric, kk))

    for k, v in evaluations.items():
        print(k)
        print(v)
    np.save(os.path.join(evaluationDir, 'cv_0704_60_collected_test_dist.npy'), evaluations)

def temp2spec(temp, is_IF=True):
    fake_s_spec = librosa.stft(temp, n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                               hop_length=conf.seeg_hop)
    s_mag = np.log(np.abs(fake_s_spec) + conf.epsilon)[: conf.w]
    s_angle = np.angle(fake_s_spec)[: conf.w]
    s_IF = phase_operation.instantaneous_frequency(s_angle, time_axis=1)
    if is_IF:
        fake_s_spec = np.stack((s_mag, s_IF), axis=0)[None, ...]
    else:
        fake_s_spec = np.stack((s_mag, s_angle), axis=0)[None, ...]
    fake_s_spec = normalizer.normalize(torch.from_numpy(fake_s_spec), 'seeg').numpy()[0]
    fake_s_mag = fake_s_spec[0]
    if is_IF:
        fake_s_IF = fake_s_spec[1]
    else:
        fake_s_IF = phase_operation.instantaneous_frequency(fake_s_spec[1], time_axis=1)

    return fake_s_mag, fake_s_IF

# tll pairs [('FZ', 'I8', array([69.34300274])), ('F4', 'A8', array([73.33140372])), ('PZ', 'E2', array([50.0013947]))]
# lk pairs [('Fp2', 'E10', array([20.69592009])), ('FZ', 'F1', array([24.33117395])), ('F4', 'F11', array([9.40280172])), ('F8', 'B14', array([14.80064537])), ('C4', 'D15', array([34.86986275])), ('T4', 'H14', array([19.30499903]))]
# zxl pairs [('CZ', 'D1', array([28.5934427])), ('PZ', 'C1', array([20.97461499])), ('O1', 'L1', array([46.05459558]))]
# yjh pairs [('F3', 'E10', array([44.71298704])), ('CZ', 'L3', array([69.89194194])), ('C3', 'G10', array([39.84213185]))]

if __name__ == '__main__':

    all_patients = ['lk', 'zxl', 'yjh', 'tll', 'lxh', 'wzw', 'lmk']
    patient = 'lk'
    home_dir = '/public/home/xlwang/hmq'
    # home_dir = '/home/hmq'
    statDir = os.path.join(home_dir, "Projects/experiments/statistics/")
    prefix = "cv"
    model_name = 'pix2pix_global_onesided'
    date = '0908'
    datasetName = 'cv_0704_60'
    experName = "_".join([prefix, model_name, date, patient])
    phase_epoch = 'test_36'
    resultDir = os.path.join(home_dir, 'Projects/experiments/results', experName, phase_epoch, 'npys')
    evaluationDir = os.path.join(home_dir, 'Infos/evaluation/')
    normalizer_dir = os.path.join(home_dir, 'Infos/norm_args/')
    method = 'GAN'  # calculate_distance的method
    is_IF = True
    is_pghi = False
    is_temporal = False
    domain = 'temporal' if is_temporal else 'freq'

    normalizer_ls = {}
    best_mapping = np.load("/public/home/xlwang/hmq/Infos/pix2pix_global_onesided_patient_best_results_mapping.npy",
                           allow_pickle=True).item()
    fake_path = []
    real_path = os.path.join(home_dir, 'Datasets', datasetName, 'A', 'test')
    # for p, path in best_mapping.items():
    #
    #     experName = path.split('/')[0]
    #     phase_epoch = path.split('/')[1][:7]
    #     fake_path.append(os.path.join(home_dir, 'Projects/experiments/results', experName, phase_epoch, 'npys'))
    #
    #     normalizer_name = datasetName + '_without_' + p
    #     if is_IF:
    #         normalizer_name += '_IF'
    #     if is_pghi:
    #         normalizer_name += '_pghi'
    #     if is_temporal:
    #         normalizer_name += '_temporal'
    #     normalizer_name += '.npy'
    #     normalizer = DataNormalizer(None, os.path.join(normalizer_dir, normalizer_name), False, use_phase=not is_pghi,
    #                                 domain=domain)
    #     normalizer_ls[p] = normalizer
    #
    # results = bestK_results(real_path, fake_path, normalizer_ls, 20, method=method, is_IF=is_IF, save_path='/public/home/xlwang/hmq/Infos/bestK/best20.npy')
    #
    # print(results)

    # jiangPicked = np.load("/home/hmq/Infos/jiang_picked.npy", allow_pickle=True).item()

    # for p in ['tll', 'yjh', 'wzw']:
    #     hdpath = os.path.join('/home/hmq/Infos/dist/', p + '_hd_by_Jiang.npy')
    #     bdpath = os.path.join('/home/hmq/Infos/dist/', p + '_bd_by_Jiang.npy')
    #     explore_eeg_seeg_dist_sim_relation(p, '/home/hmq/Infos/dist/' + p + '_pos_dist_by_Jiang.npy', bdpath, hdpath,
    #                                        saveDir='/home/hmq/Infos/distSimCorr/allEEGallSEEG')
        # hd = np.load(hdpath, allow_pickle=True).item()
        # bd = np.load(bdpath, allow_pickle=True).item()
        # pos_dist = np.load('/home/hmq/Infos/dist/' + p + '_pos_dist.npy', allow_pickle=True).item()
        # for k in pos_dist.keys():
        #     new_hd = []
        #     for pair in pos_dist[k]:
        #         if pair[0] in jiangPicked[p]:
        #             new_hd.append(pair)
        #     print('before: {}, after: {}'.format(len(pos_dist[k]), len(new_hd)))
        #     pos_dist[k] = new_hd
        # np.save('/home/hmq/Infos/dist/' + p + '_pos_dist_by_Jiang.npy', pos_dist)

    # raw = mne.io.read_raw_edf("/home/hmq/Signal/LK_Sleep_Aug_4th_2am.edf")
    # raw.pick_channels(['POL D13'])
    # raw.crop(tmin=0, tmax=0 + 1016 / 64.)
    # raw.resample(conf.seeg_sf, npad="auto")
    # raw.filter(1., None, fir_design='firwin')
    # filtered_data = raw.get_data()[0]
    # fig, ax = plt.subplots(1, 1)
    # ax.set_ylim([-0.0015, 0.0015])
    # ax.plot(np.arange(filtered_data.size), filtered_data)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    # baseline = np.load("/public/home/xlwang/hmq/Infos/evaluation/cv_0704_60_EEG_avg_test_dist_v2.npy", allow_pickle=True).item()
    # results = np.load("/public/home/xlwang/hmq/Infos/evaluation/cv_0704_60_asae_avg_test_dist_v1.npy", allow_pickle=True).item()
    # print(results)
    # for k, v in baseline.items():
    #     metric = k.split('_')[0]
    #     if 'mean' in k:
    #         baseline_mean = v
    #         try:
    #             baseline_std = baseline[metric + '_std']
    #         except:
    #             pass
    #     else:
    #         continue
    #     if k in results.keys():
    #         try:
    #             results[k] *= baseline_mean ** -1
    #             results[k] = np.log2(results[k])
    #             results[metric + '_std'] *= baseline_std ** -1
    #             results[metric + '_std'] = np.log2(results[metric + '_std'])
    #         except:
    #             pass
    normalizer = DataNormalizer(None, os.path.join(normalizer_dir, 'cv_0704_60_without_lk_IF.npy'), False, use_phase=not is_pghi,
                                                                domain=domain)
    realeeg = np.load('/public/home/xlwang/hmq/Datasets/cv_0704_60/B/test/lk/lk_CZ_D12_1277.npy')[0][:1016]
    realseeg = np.load('/public/home/xlwang/hmq/Datasets/cv_0704_60/A/test/lk/lk_CZ_D12_1277.npy')[0][:1016]
    realeegspec = np.load(
        '/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_CZ_D12_1277_real_A.npy')
    realseegspec = np.load(
        '/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_CZ_D12_1277_real_B.npy')
    fakeseegspec = np.load('/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_CZ_D12_1277_fake_B.npy')
    fakegansynthspec = np.load('/public/home/xlwang/hmq/Projects/experiments/results/cv_gansynth_0905_lk/test_70/npys/lk_CZ_D12_1277_fake_B.npy')
    fakeIFspec = np.load('/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_IF_0913_lk/test_50/npys/lk_CZ_D12_1277_fake_B.npy')
    fseeg_temp = IF_to_eeg(fakeseegspec, normalizer, iseeg=False, is_IF=is_IF)[0]
    fakeseeg = mne.filter.filter_data(fseeg_temp, 64, 0.5, None)
    fakemag, fakeIF = temp2spec(fakeseeg)
    fseeg_temp_IF = IF_to_eeg(fakeIFspec, normalizer, iseeg=False, is_IF=is_IF)[0]
    fakeIFtemp = mne.filter.filter_data(fseeg_temp_IF, 64, 0.5, None)
    fakeIFmag, fakeIFIF = temp2spec(fakeIFtemp)
    fseeg_temp_gs = IF_to_eeg(fakegansynthspec, normalizer, iseeg=False, is_IF=is_IF)[0]
    fakegansynthtemp = mne.filter.filter_data(fseeg_temp_gs, 64, 0.5, None)
    fakegsmag, fakegsIF = temp2spec(fakegansynthtemp)

    title_list = ['Input', 'IF', 'GANSynth', 'Ours', 'Groundtruth']
    xlabel = 'Time'
    ylabel = 'Frequency'
    # plotting_list = [realeeg, fakeIFtemp, fakegansynthtemp, fakeseeg, realseeg]
    mag_list = [realeegspec[0], fakeIFmag, fakegsmag, fakemag, realseegspec[0]]
    IF_list = [realeegspec[1], fakeIFIF, fakegsIF, fakeIF, realseegspec[1]]
    fig = plt.figure()

    # for i, to_plot in enumerate(plotting_list):
    #     ax = fig.add_subplot(len(plotting_list), 1, i + 1)
    #     ax.set_title(title_list[i])
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     x = np.array([i for i in range(1016)])
    #     ax.plot(x, to_plot, linewidth=0.6)
    import matplotlib.gridspec as gridspec
    # grids = gridspec.GridSpec(2, 5)
    # fig, axes = plt.subplots(2, len(mag_list))
    fig = plt.figure(constrained_layout=True)
    widths = [1 for _ in range(5)]
    heights = [1 for _ in range(5)]
    grids = fig.add_gridspec(ncols=5, nrows=2)

    for i in range(len(mag_list)):
        # ax = fig.add_subplot(2, len(mag_list), i + 1)
        ax = fig.add_subplot(grids[0, i])
        ax.set_title(title_list[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        # x = np.array([i for i in range(1016)])
        times = np.array([i for i in range(len(fakemag[0]))])
        mesh = ax.pcolormesh(times, times, mag_list[i],
                             cmap='RdBu_r', vmin=-1, vmax=1)
        # axes[0][i].set(ylim=freqs[[0, -1]], xlabel=xlabel, ylabel=ylabel)
        # fig.colorbar(mesh, ax=ax)

        ax1 = fig.add_subplot(grids[1, i])
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_xticks([])
        ax1.set_yticks([])
        mesh = ax1.pcolormesh(times, times, IF_list[i],
                                     cmap='RdBu_r')
        # axes[0][i].set(ylim=freqs[[0, -1]], xlabel=xlabel, ylabel=ylabel)
        # fig.colorbar(mesh, ax=ax1)
        # axes[0][i].plot(x, to_plot, linewidth=0.6)

    plt.tight_layout()
    plt.show()
