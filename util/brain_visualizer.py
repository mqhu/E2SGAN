import mne
import os
import numpy as np
from util.eeg_tools import read_raw_signal, exclusions, eeg_ch, seeg_ch
from mat_file import retrieve_chs_from_mat
import matplotlib.pyplot as plt
import util.numpy_tools as nutil
from GANMetrics.metric import get_correlation_mat, compare_correlation_array, print_score
from gansynth.normalizer import DataNormalizer


def plot_positions(pos_dict, kind='topomap', azim=45, elev=45, show=True, savePath=None, title=""):
    '''
    Plot topomap or 3d map of electodes
    :param pos_dict: dictionary of electrode positions in meters
    :param kind: string in ['topomap', '3d']
    :param azim: azimuth angle in the x,y plane (in degrees)
    :param elev: elevation angle in the z plane (in degrees)
    :return: None
    '''
    sample_path = mne.datasets.sample.data_path(download=False)
    subject = 'fsaverage'
    subjects_dir = sample_path + '/subjects'
    lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
        subject, subjects_dir=subjects_dir)
    lpa = [x / 1000 for x in lpa['r']]
    nasion = [x / 1000 for x in nasion['r']]
    rpa = [x / 1000 for x in rpa['r']]
    montage = mne.channels.make_dig_montage(pos_dict, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='mri')
    fig = montage.plot(scale_factor=10, kind=kind, show_names=True)
    fig.suptitle(title)
    plt.tight_layout()

    if kind == '3d':
        fig.gca().view_init(azim=azim, elev=elev)
    if show:
        plt.show()
    elif savePath is not None:
        plt.savefig(savePath)


def plot_topomap(data, info, show=True, ax=None):
    '''
    Plot topomap according to given data and channel info
    :param data: data to plot
    :param info: mne.info containing channel position
    :param ch_idx: indices of channels to be plotted
    :return: None
    '''
    if ax is None:
        im, _ = mne.viz.plot_topomap(data, info, show=False, vmin=0, vmax=1)
    else:
        im, _ = mne.viz.plot_topomap(data, info, show=False, axes=ax)
    plt.colorbar(im)
    if show:
        plt.show()
    else:
        return im


def visualize_correlation_topomap(e_band, s_band, show=True):
    seeg_ds = nutil.make_dataset("/home/cbd109/Users/hmq/GANDatasets/LK_rest/A/val/")
    eeg_ds_r = nutil.make_dataset("/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/val/")
    eeg_ds_ff = nutil.make_dataset("/home/cbd109/Users/hmq/codes/pix2pix/results/IF_GAN/val_latest/npys/")
    normalizer = DataNormalizer(None, False)
    eeg_ds_f = []
    for f_n in eeg_ds_ff:
        if f_n.find('fake') > 0:
            eeg_ds_f.append(f_n)
    eeg_seeg_pairs = np.load("/home/cbd109/Users/hmq/LK_info/eeg_seeg_pairs_all_eeg.npy", allow_pickle=True).item()[
        'pairs']
    inclusive_eeg = []
    for ch in eeg_ch:
        if ch not in exclusions:
            inclusive_eeg.append(ch)
    order = []
    for pair in eeg_seeg_pairs:
        try:
            eeg_idx = inclusive_eeg.index(pair[0])
            seeg_idx = seeg_ch.index(pair[1])
        except:
            continue
        order.append((eeg_idx, seeg_idx))
    corr_r = get_correlation_mat(seeg_ds, eeg_ds_r, 130, order)
    corr_f = get_correlation_mat(seeg_ds, eeg_ds_f, 130, order, is_fake=True, normalizer=normalizer)
    valid_idx = compare_correlation_array(corr_r, corr_f, 120, 120)[-1]

    top_k = []
    k = 5
    for i in range(144):
        count = sum([valid_idx[j] for j in range(i, 144 * len(order), 144)])
        top_k.append((count, i))
    top_k = sorted(top_k)[-k:]
    freq_idx = [pair[1] for pair in top_k]
    print(freq_idx)
    e_bands = [i % 12 for i in freq_idx]
    s_bands = [i // 12 for i in freq_idx]
    print(e_bands)
    print(s_bands)

    picked_eeg = {}
    for idx_pair in order:
        ch_name = inclusive_eeg[idx_pair[0]]
        picked_eeg[ch_name] = np.asarray(LK_eeg_pos[ch_name]) / 1000.
    montage = mne.channels.make_dig_montage(picked_eeg)
    info = mne.create_info(ch_names=list(picked_eeg.keys()), ch_types=['eeg' for _ in range(len(order))], sfreq=128.)
    info.set_montage(montage)
    corr_data_r = []
    for i in range(len(order)):
        v = corr_r[i * 144 + s_band * 12 + e_band]
        corr_data_r.append(abs(v))
    corr_data_f = []
    for i in range(len(order)):
        v = corr_f[i * 144 + s_band * 12 + e_band]
        corr_data_f.append(abs(v))
    #fig, axes = plt.subplots(2, 1)
    #axes[0].set_title('Real')
    #axes[1].set_title('Fake')
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    #ax0.title.set_text('Real')
    #ax1.title.set_text('Fake')
    im1, _ = mne.viz.plot_topomap(corr_data_r, info, show=False, axes=ax0, vmin=0, vmax=1)
    im2, _ = mne.viz.plot_topomap(corr_data_f, info, show=False, axes=ax1, vmin=0, vmax=1)
    #fig.subplots_adjust(right=0.6, left=0.1)
    #cbar_ax = fig.add_axes([0.62, 0.8, 0.02, 0.4])
    #fig.colorbar(im2, ax=cbar_ax)
    plt.subplots_adjust(top=0.5)
    fig.colorbar(im1, ax=ax0)
    fig.colorbar(im2, ax=ax1)
    plt.title(' ', y=-0.3)
    plt.tight_layout()
    if show:
        plt.show()
    '''if isreal:
        corr_data_r = []
        for i in range(len(order)):
            v = corr_r[i * 144 + s_band * 12 + e_band]
            corr_data_r.append(abs(v))
        plot_topomap(corr_data_r, info, show=True)
    else:
        corr_data_f = []
        for i in range(len(order)):
            v = corr_f[i * 144 + s_band * 12 + e_band]
            corr_data_f.append(abs(v))
        plot_topomap(corr_data_f, info, show=True)'''
    #fig, ax = plt.subplots(2, 1)
    #im_r = plot_topomap(corr_data_r, info, show=False, ax=ax[0])
    #im_f = plot_topomap(corr_data_f, info, show=False, ax=ax[1])


if __name__ == '__main__':

    # LK_eeg_pos = np.load("/home/cbd109/Users/hmq/LK_info/LK_eeg_coords.npy", allow_pickle=True).item()
    # seeg_info = np.load("/home/cbd109/Users/hmq/LK_info/LK_name_pos_wc.npy", allow_pickle=True).tolist()
    patient = 'lmk'
    eeg_pos = np.load("/home/hmq/Infos/position_info/eeg_pos.npy", allow_pickle=True).item()
    seeg_pos = np.load("/home/hmq/Infos/position_info/" + patient + "_seeg_pos.npy", allow_pickle=True).tolist()
    eeg_names = np.load("/home/hmq/Infos/ch_names/" + patient + "_stripped_eeg.npy", allow_pickle=True)
    pickedSEEG = np.load("/home/hmq/Infos/jiang_picked.npy", allow_pickle=True).item()[patient]
    picked_pos = {}
    for d in seeg_pos:
        if d['name'] in ['A14', 'H14']:
            # picked_pos[d['name']] = list(d['pos'])
            picked_pos[d['name']] = [x / 1000 for x in d['pos']]
    # print(picked_pos)
    # print(len(picked_pos))

    for ch, pos in eeg_pos.items():
        if ch in ['CZ']:
            # picked_pos['EEG ' + ch] = list(pos)
            picked_pos['EEG ' + ch] = [x / 1000 for x in pos]

    # print(picked_pos)
    # print(len(picked_pos))

    plot_positions(picked_pos, kind='3d', show=False, savePath='/home/hmq/' + patient + '_3d_lat.png', azim=0, elev=0)
    plot_positions(picked_pos, kind='3d', show=False, savePath='/home/hmq/' + patient + '_3d_front.png', azim=90, elev=0)
    plot_positions(picked_pos, kind='topomap', show=False, savePath='/home/hmq/' + patient + '_topo.png', azim=90, elev=0)


    # eeg_ch_names = LK_eeg_pos.keys()
    # seeg_ch_names = [ch['name'] for ch in seeg_info]
    # all_pos = {}
    # for k, v in LK_eeg_pos.items():
    #     if k not in exclusions:
    #         all_pos[k] = np.asarray([i / 1000. for i in v])
    # for dic in seeg_info:
    #     all_pos[dic['name']] = dic['pos'] / 1000.
    # #plot_positions(all_pos, kind='3d', show=False, elev=30)
    # visualize_correlation_topomap(3, 11, False)
    # plt.savefig('full_corr_alike.tif', dpi=600)
