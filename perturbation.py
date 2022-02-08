import numpy as np
import torch
from scipy.stats import pearsonr
from util.distance_metrics import bestK_results
from gansynth.normalizer import DataNormalizer
import os
from options.test_options import TestOptions
from models import create_model
from util import html
from util.numpy_tools import save_origin_npy
import matplotlib.pyplot as plt
from data.eeg_dataset import EEGDatasetDataloader
import shutil
from util.eeg_tools import IF_to_eeg
import mne
from dtaidistance import dtw

EEGTrainingDir = '/public/home/xlwang/hmq/Datasets/cv_0704_60/B/train'
SEEGTrainingDir = '/public/home/xlwang/hmq/Datasets/cv_0704_60/A/train'
resultDir = '/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_tll/trainAsTest_36/npys'
normalizerDir = '/public/home/xlwang/hmq/Infos/norm_args/cv_0704_60_without_tll_IF.npy'
EEGPosDir = '/public/home/xlwang/hmq/Infos/position_info/eeg_pos.npy'
patientList = ['lk', 'zxl', 'yjh', 'lxh', 'wzw', 'lmk']  # no tll
freq_bands = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 16), 'beta': (16, 32)}

# def searchTopK(topK, aggregateLoc=True, save_path=None):
#     normalizer = DataNormalizer(None, normalizerDir, False, use_phase=True, domain='freq')
#     topKresults = bestK_results(SEEGTrainingDir, resultDir, [normalizer], topK, is_IF=True, aggregateLoc=aggregateLoc, save_path=save_path)
#

def frequencywise_perturbation(original, low, high):

    freq_step = 128 // 32
    width = original.shape[0]
    std = original.std()
    gaussianNoise = np.random.normal(loc=0, scale=std, size=((high - low) * freq_step, width))
    original[low * freq_step: high * freq_step, :] += gaussianNoise

    return original, gaussianNoise.mean()


def visualize_correlation_topomap(correlations, saveDir=None):

    eeg_pos = np.load(EEGPosDir, allow_pickle=True).item()
    picked_eeg = {}
    for chan in correlations['delta'].keys():
        picked_eeg[chan] = np.asarray(eeg_pos[chan]) / 1000.

    montage = mne.channels.make_dig_montage(picked_eeg)
    info = mne.create_info(ch_names=list(picked_eeg.keys()), ch_types=['eeg' for _ in range(len(picked_eeg))], sfreq=64.)
    info.set_montage(montage)

    delta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['delta'].items()]
    theta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['theta'].items()]
    alpha_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['alpha'].items()]
    beta_corr = [np.mean(corr_ls) for chan, corr_ls in correlations['beta'].items()]

    fig = plt.figure()
    ax0 = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)
    ax0.title.set_text('Delta')
    ax1.title.set_text('Theta')
    ax2.title.set_text('Alpha')
    ax3.title.set_text('Beta')
    im0, _ = mne.viz.plot_topomap(delta_corr, info, show=False, axes=ax0, vmin=0, vmax=1)
    im1, _ = mne.viz.plot_topomap(theta_corr, info, show=False, axes=ax1, vmin=0, vmax=1)
    im2, _ = mne.viz.plot_topomap(alpha_corr, info, show=False, axes=ax2, vmin=0, vmax=1)
    im3, _ = mne.viz.plot_topomap(beta_corr, info, show=False, axes=ax3, vmin=0, vmax=1)
    #fig.subplots_adjust(right=0.6, left=0.1)
    #cbar_ax = fig.add_axes([0.62, 0.8, 0.02, 0.4])
    #fig.colorbar(im2, ax=cbar_ax)
    # plt.subplots_adjust(top=0.5)
    # fig.colorbar(im1, ax=ax0)
    # fig.colorbar(im2, ax=ax1)
    cbar_ax = fig.add_axes([0.81, 0.35, 0.02, 0.33])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.ax.set_ylabel("Correlation")

    # plt.title(' ', y=-0.3)
    plt.tight_layout()
    if saveDir is not None:
        plt.savefig(os.path.join(saveDir, 'perturbationCorrelation'))
    else:
        plt.show()


if __name__ == '__main__':

    topK = 100
    n_perturbation = 50
    # investigated = 'lk'
    aggregateLoc = False
    perturbation_saveDir = '/public/home/xlwang/hmq/Infos/perturbation'
    normalizer = DataNormalizer(None, normalizerDir, False, use_phase=True, domain='freq')
    # normalizers = {}
    # for p in patientList:
    #     normalizers[p] = normalizer
    # topKresults = bestK_results(SEEGTrainingDir, [resultDir], normalizers, topK, is_IF=True, aggregateLoc=aggregateLoc, saveDir=perturbation_saveDir)
    # print(topKresults)

    correlations = dict.fromkeys(list(freq_bands.keys()), 0)
    pvalue = dict.fromkeys(list(freq_bands.keys()), 0)
    for k in correlations.keys():
        correlations[k] = {}
        pvalue[k] = {}

    parser = TestOptions()  # get training options
    opt = parser.parse(save_opt=False)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    experiment_name_prefix = opt.name
    ae_prefix = opt.ae_name
    opt.name = experiment_name_prefix + '_' + opt.leave_out
    opt.ae_name = ae_prefix + '_' + opt.leave_out
    parser.save_options(opt)
    dataset_name = os.path.basename(opt.dataroot)

    dataset = EEGDatasetDataloader(opt, patientList)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.set_normalizer(normalizer)
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    npy_dir = os.path.join(web_dir, "npys")
    if os.path.exists(npy_dir):
        shutil.rmtree(npy_dir)
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    print('Dataset size:{}'.format(len(dataset)))

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        f_n = os.path.basename(data['A_paths'][0])
        f_n = f_n.split('.')[0]
        f_n = f_n + '_fake_B.npy'
        patient = f_n.split('_')[0]
        EEG_chan = f_n.split('_')[1]
        # if patient != investigated:
        #     continue
        for band_name in freq_bands.keys():
            if EEG_chan not in correlations[band_name].keys():
                correlations[band_name][EEG_chan] = []
                pvalue[band_name][EEG_chan] = []

        real = data['A'][0].numpy()
        real = IF_to_eeg(real, normalizer, iseeg=False, is_IF=True)[0].astype(np.float64)
        real_psd = mne.time_frequency.psd_array_welch(real, 64, n_fft=256)
        original_fake = np.load(os.path.join(resultDir, f_n))
        original_fake = IF_to_eeg(original_fake, normalizer, iseeg=False, is_IF=True)[0].astype(np.float64)
        original_psd = mne.time_frequency.psd_array_welch(original_fake, 64, n_fft=256)
        original_psd_dist = np.linalg.norm(np.asarray(real_psd[0]) - np.asarray(original_psd[0]), ord=2) / len(real_psd[0]) ** 0.5
        # original_dtw = dtw.distance_fast(real, original_fake, use_pruning=True)

        for band_name, band in freq_bands.items():
            mag_increment_ls = []
            psd_increment_ls = []
            # dtw_increment_ls = []
            for i in range(n_perturbation):
                perturbated, mag_increment = frequencywise_perturbation(data['B'][0][0].numpy(), band[0], band[1])  # perturbate input EEG
                data['B'][0][0] = torch.from_numpy(perturbated)
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results 模仿一下这一段把原始的numpy数据给保存下来
                perturbated_fake = visuals['fake_B'][0].cpu().numpy()
                perturbated_fake = IF_to_eeg(perturbated_fake, normalizer, iseeg=False, is_IF=True)[0].astype(np.float64)
                perturbated_psd = mne.time_frequency.psd_array_welch(perturbated_fake, 64, n_fft=256)
                perturbated_psd_dist = np.linalg.norm(np.asarray(real_psd[0]) - np.asarray(perturbated_psd[0]), ord=2) / len(real_psd[0]) ** 0.5
                psd_increment = perturbated_psd_dist - original_psd_dist
                # perturbated_dtw = dtw.distance_fast(real, perturbated_fake, use_pruning=True)
                # dtw_increment = perturbated_dtw - original_dtw
                mag_increment_ls.append(mag_increment)
                psd_increment_ls.append(psd_increment)
                # dtw_increment_ls.append(dtw_increment)

            corr1, p1 = pearsonr(mag_increment_ls, psd_increment_ls)
            # corr2, p2 = pearsonr(mag_increment_ls, dtw_increment_ls)
            correlations[band_name][EEG_chan].append(corr1)
            # correlations[band_name][EEG_chan][1].append(corr2)
            pvalue[band_name][EEG_chan].append(p1)
            # pvalue[band_name][EEG_chan][1].append(p2)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... ' % i)

    np.save(os.path.join(perturbation_saveDir, 'perturbationCorrPSD'), correlations)
    np.save(os.path.join(perturbation_saveDir, 'perturbationPvaluePSD'), pvalue)
    # webpage.save()  # save the HTML