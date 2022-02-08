import mne.io
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from util.distance_metrics import compare_by_interval, psd_probability, Hellinger_Distance
from gansynth.normalizer import DataNormalizer
from util.numpy_tools import plot_hist_distribution, plot_spectrogram
from dtaidistance import dtw
from util.brain_visualizer import plot_positions

if __name__ == '__main__':

    plt.rc('font', family='Times New Roman')
    # home_dir = '/public/home/xlwang/hmq'
    home_dir = '/home/hmq'
    datasetName = 'cv_0704_60'
    normalizer_dir = os.path.join(home_dir, 'Infos/norm_args/')
    # p_eval = np.load(os.path.join(home_dir, 'Infos/pix2pix_global_patient_best_results_mapping.npy'),
    #                  allow_pickle=True).item()
    is_IF = True
    is_pghi = False
    is_temporal = False

    # stds = dict()
    # for itv in range(2, 508 + 1):
    #
    #     print('itv:', itv)
    #     results = []
    #     for p in p_eval.keys():
    #         experName = p_eval[p].split('/')[0]
    #         phase_epoch = p_eval[p].split('/')[1][:7]
    #         normalizer_name = datasetName + '_without_' + p
    #         if is_IF:
    #             normalizer_name += '_IF'
    #         if is_pghi:
    #             normalizer_name += '_pghi'
    #         if is_temporal:
    #             normalizer_name += '_temporal'
    #         normalizer_name += '.npy'
    #         domain = 'temporal' if is_temporal else 'freq'
    #         normalizer = DataNormalizer(None, os.path.join(normalizer_dir, normalizer_name), False,
    #                                     use_phase=not is_pghi,
    #                                     domain=domain)
    #         real_path = os.path.join(home_dir, 'Datasets', datasetName, 'A', phase_epoch.split('_')[0])
    #         fake_path = os.path.join(home_dir, 'Projects/experiments/results', experName, phase_epoch, 'npys')
    #         result = compare_by_interval(real_path, fake_path, normalizer, itv, method='GAN', is_IF=is_IF, aggregate=False, save_path=None)
    #         results.append(result)
    #
    #     rmse_interval_mean = 0
    #     for i, p in enumerate(p_eval.keys()):
    #         score = results[i][p]
    #         rmse_interval_mean += score['rmse_interval_mean']
    #         if i == 6:
    #             rmse_interval_mean /= 7
    #
    #     rmse_std = rmse_interval_mean.std()
    #     stds[itv] = rmse_std
    #     print('rmse_std:', rmse_std)
    # print(stds)
    # np.save(os.path.join(home_dir, 'Infos/evaluation/', '_'.join([datasetName, 'pix2pix_global_rmse_distribution'])), stds)

    '''std distribution'''
    '''
    glb_one_rmse_dist = np.load(os.path.join(home_dir, 'Infos/evaluation/', 'cv_0704_60_pix2pix_global_onesided_rmse_distribution.npy'), allow_pickle=True).item()
    glb_rmse_dist = np.load(os.path.join(home_dir, 'Infos/evaluation/', 'cv_0704_60_pix2pix_global_rmse_distribution.npy'), allow_pickle=True).item()
    IF_rmse_dist = np.load(os.path.join(home_dir, 'Infos/evaluation/', 'cv_0704_60_pix2pix_IF_rmse_distribution.npy'), allow_pickle=True).item()
    glb_one_rmse_dist = np.asarray(list(glb_one_rmse_dist.values()))
    glb_rmse_dist = np.asarray(list(glb_rmse_dist.values()))
    IF_rmse_dist = np.asarray(list(IF_rmse_dist.values()))
    sns.set_style('white')
    
    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=(8, 4))  # 绘制画布
    sns.distplot(glb_one_rmse_dist, hist=False, bins=250, kde=True, rug=False,
                 rug_kws={'color': 'b', 'lw': 2, 'alpha': 0.5, 'height': 0.1},  # 设置数据频率分布颜色#控制是否显示观测的小细条（边际毛毯）
                 kde_kws={"color": "b", "lw": 2, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
                 label='E2SGAN')
    sns.distplot(glb_rmse_dist, hist=False, bins=250, kde=True, rug=False,
                 rug_kws={'color': 'g', 'lw': 2, 'alpha': 0.5, 'height': 0.1},  # 设置数据频率分布颜色#控制是否显示观测的小细条（边际毛毯）
                 kde_kws={"color": "g", "lw": 2, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
                 label='w/o CSA')
    sns.distplot(IF_rmse_dist, hist=False, bins=250, kde=True, rug=False,
                 rug_kws={'color': 'r', 'lw': 2, 'alpha': 0.5},
                 kde_kws={"color": "r", "lw": 2, 'linestyle': '--'},
                 label='w/o CSA & WPP')
    plt.axvline(glb_one_rmse_dist.mean(), color='b', linestyle=":", alpha=0.8)
    plt.axvline(glb_rmse_dist.mean(), color='g', linestyle=":", alpha=0.8)
    plt.axvline(IF_rmse_dist.mean(), color='r', linestyle=":", alpha=0.8)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    plt.grid(linestyle='--')  # 添加网格线
    plt.legend()
    plt.xlabel('Standard Deviation')
    plt.title('std(RMSE) Distribution')
    plt.show()
    '''
    # real_eeg= np.load("/public/home/xlwang/hmq/Datasets/cv_0704_60/B/test/yjh/yjh_C3_F13_2341.npy", allow_pickle=True)[0]
    # real_seeg = np.load("/public/home/xlwang/hmq/Datasets/cv_0704_60/A/test/yjh/yjh_C3_F13_2341.npy", allow_pickle=True)[0]
    # eeg_mag = np.load("/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_yjh/test_32/npys/yjh_C3_F13_2341_real_A.npy", allow_pickle=True)[0]
    # seeg_mag = np.load("/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_yjh/test_32/npys/yjh_C3_F13_2341_real_B.npy", allow_pickle=True)[0]
    # # fig, ax = plt.subplots(2,1)
    # # ax[0].plot(np.arange(len(real_eeg)), real_eeg)
    # # ax[1].plot(np.arange(len(real_seeg)), real_seeg)
    # plot_spectrogram(np.arange(128), np.arange(128), vmin=-1, vmax=1, EEG=eeg_mag, SEEG=seeg_mag)
    # plt.show()

    patient = 'lmk'

    eeg = mne.io.read_raw_fif(os.path.join(home_dir, 'Signal/preprocessed/lmk_eeg_raw.fif'))
    seeg = mne.io.read_raw_fif(os.path.join(home_dir, 'Signal/preprocessed/lmk_seeg_raw.fif'))
    eeg_pos = np.load(os.path.join(home_dir, "Infos/position_info/eeg_pos.npy"), allow_pickle=True).item()
    seeg_pos = np.load(os.path.join(home_dir, "Infos/position_info/" + patient + "_seeg_pos.npy"), allow_pickle=True).tolist()
    # cz = eeg.pick_channels(['CZ']).get_data()[0]
    # cz_pos = np.asarray(eeg_pos['CZ'])
    # cz_psd = mne.time_frequency.psd_array_welch(cz, 64, n_fft=256)
    # cz_prob = psd_probability(cz_psd[0], 256, cz.shape[0])
    # min_sim = {'s_name':'', 'hd': float('inf'), 'pos': [], 'dist': 0}
    # max_sim = {'s_name':'', 'hd': 0, 'pos': [], 'dist': 0}
    # a14 = seeg.copy().pick_channels(['A14']).get_data()[0]
    # a14_psd = mne.time_frequency.psd_array_welch(a14, 64, n_fft=256)
    # a14_prob = psd_probability(a14_psd[0], 256, a14.shape[0])
    # a14_psd_dist = np.linalg.norm(np.asarray(cz_psd[0]) - np.asarray(a14_psd[0]), ord=2) / len(cz_psd[0]) ** 0.5
    # a14_dtw_dist = dtw.distance_fast(cz, a14, use_pruning=True)
    # a14_hd = Hellinger_Distance(cz_prob, a14_prob)
    # h14 = seeg.pick_channels(['H14']).get_data()[0]
    # h14_psd = mne.time_frequency.psd_array_welch(h14, 64, n_fft=256)
    # h14_prob = psd_probability(h14_psd[0], 256, h14.shape[0])
    # h14_psd_dist = np.linalg.norm(np.asarray(cz_psd[0]) - np.asarray(h14_psd[0]), ord=2) / len(cz_psd[0]) ** 0.5
    # h14_dtw_dist = dtw.distance_fast(cz, h14, use_pruning=True)
    # h14_hd = Hellinger_Distance(cz_prob, h14_prob)
    # print('A14:')
    # print('hd:{}, psd:{}, dtw:{}'.format(a14_hd, a14_psd_dist, a14_dtw_dist))
    # print('H14:')
    # print('hd:{}, psd:{}, dtw:{}'.format(h14_hd, h14_psd_dist, h14_dtw_dist))
    # np.save('/home/hmq/' + patient + '_dist_sim_comparison.npy',
    #         {'A14': 'hd:{}, psd:{}, dtw:{}'.format(a14_hd, a14_psd_dist, a14_dtw_dist),
    #          'H14': 'hd:{}, psd:{}, dtw:{}'.format(h14_hd, h14_psd_dist, h14_dtw_dist)})

    '''可视化对比图，对比距离近但相似度远'''
    picked_pos = {}
    for d in seeg_pos:
        if d['name'] in ['A14', 'H14']:
            # picked_pos[d['name']] = list(d['pos'])
            picked_pos['SEEG ' + d['name']] = [x / 1000 for x in d['pos']]
        if d['name'] == 'A14':
            picked_pos['SEEG ' + d['name']] = np.asarray(picked_pos['SEEG ' + d['name']]) - np.asarray([0, 0.007, 0])

    for ch, pos in eeg_pos.items():
        if ch in ['CZ']:
            # picked_pos['EEG ' + ch] = list(pos)
            picked_pos['EEG ' + ch] = [x / 1000 for x in pos]

    plt.rcParams.update({"font.size": 12})

    # plot_positions(picked_pos, kind='topomap', show=False, savePath='/home/hmq/' + patient + '_topo.pdf', azim=90, elev=0, title='')
    # plot_positions(picked_pos, kind='3d', show=False, savePath='/home/hmq/' + patient + '_3d_front.png', azim=90, elev=0)
    plot_positions(picked_pos, kind='3d', show=False, savePath='/home/hmq/' + patient + '_3d.pdf', azim=45, elev=30, title='')

    # for d in seeg_pos:
    #     name = d['name']
    #     pos = d['pos']
    #     type = d['type']
    #     dist = np.linalg.norm(cz_pos - pos)
    #     if type != 'ctx' or 20 <= dist < 60 or dist > 65:
    #         continue
    #
    #     s_data = seeg.copy().pick_channels([name]).get_data()[0]
    #     s_psd = mne.time_frequency.psd_array_welch(s_data, 64, n_fft=256)
    #     s_prob = psd_probability(s_psd[0], 256, s_data.shape[0])
    #     hd = Hellinger_Distance(cz_prob, s_prob)
    #     if dist <= 20 and hd > max_sim['hd']:
    #         max_sim['s_name'] = name
    #         max_sim['hd'] = hd
    #         max_sim['pos'] = pos
    #         max_sim['dist'] = dist
    #     if 60 <= dist <= 65 and hd <min_sim['hd']:
    #         min_sim['s_name'] = name
    #         min_sim['hd'] = hd
    #         min_sim['pos'] = pos
    #         min_sim['dist'] = dist
    #     del s_data
    #
    # print('min_sim:', min_sim)
    # print('max_sim:', max_sim)


    # r_psd = mne.time_frequency.psd_array_welch(real, conf.seeg_sf, fmin=bands[-1][0], fmax=bands[-1][1],
    #                                            n_fft=conf.seeg_n_fft)
    # f_psd = mne.time_frequency.psd_array_welch(fake, conf.seeg_sf, fmin=bands[-1][0], fmax=bands[-1][1],
    #                                            n_fft=conf.seeg_n_fft)
    # dist = np.linalg.norm(np.asarray(r_psd[0]) - np.asarray(f_psd[0]), ord=2) / len(r_psd[0]) ** 0.5
    # patient_result['psd_dist'][patient][EEG_chan].append(dist)
    #
    # # 计算相位/IF的海明距离
    # r_psd_prob = psd_probability(r_psd[0], conf.seeg_n_fft, real.shape[0])
    # f_psd_prob = psd_probability(f_psd[0], conf.seeg_n_fft, real.shape[0])
    # hd = Hellinger_Distance(r_psd_prob, f_psd_prob)
    # bd = Bhattacharyya_Distance(r_psd_prob, f_psd_prob)
    # patient_result['hd'][patient][EEG_chan].append(hd)
    # patient_result['bd'][patient][EEG_chan].append(bd)
    # grouped_files[patient][EEG_chan].append(f_name)
