'''
画IF和MAG的混合
    # realspec = np.load("/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_FZ_D13_2737_real_B.npy", allow_pickle=True)
    fakespec = np.load("/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_F3_F8_107_fake_B.npy", allow_pickle=True)
    # IFspec = np.load(
    #     "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_IF_0913_lk/test_50/npys/lk_FZ_D13_2737_fake_B.npy",
    #     allow_pickle=True)
    # fakespec_1 = np.load(
    #     "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_28/npys/lk_FZ_D13_2505_fake_B.npy",
    #     allow_pickle=True)
    # r_query = np.load(
    #     "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/attention/lk_CZ_D12_2528_r_query.npy",
    #     allow_pickle=True)
    # r_key = np.load(
    #     "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/attention/lk_CZ_D12_2528_r_key.npy",
    #     allow_pickle=True)
    f_query = np.load(
        "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/attention/lk_F3_F8_107_f_query.npy",
        allow_pickle=True)
    f_key = np.load(
        "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/attention/lk_F3_F8_107_f_key.npy",
        allow_pickle=True)
    eegspec = np.load(
        "/public/home/xlwang/hmq/Projects/experiments/results/cv_pix2pix_global_onesided_0908_lk/test_24/npys/lk_F3_F8_107_real_A.npy",
        allow_pickle=True)

    bound = 100
    # realIFmean = np.mean(realspec[1], axis=1)[:, None]
    # realIFstd = np.std(realspec[1], axis=1)[:, None]
    # fakeIFmean = np.mean(fakespec[1], axis=1)[:, None]
    # fakeIFstd = np.std(fakespec[1], axis=1)[:, None]
    # eegIFmean = np.mean(eegspec[1], axis=1)[:, None]
    # eegIFstd = np.std(eegspec[1], axis=1)[:, None]
    # print('mean shape:', realIFmean.shape)
    # print('std shape:', realIFstd.shape)
    # mixedIF1 = np.concatenate((realspec[1][:bound, :], fakespec[1][bound:, :]), axis=0)
    # mixedIF1 = np.concatenate((realspec[1][:, :0], fakespec[1][:, :30], realspec[1][:, 30:]), axis=1)
    # mixedIF2 = np.concatenate((fakespec[1][:bound, :], realspec[1][bound:, :]), axis=0)
    # mixedIF1 = np.concatenate((realspec[1][:, :5], fakespec[1][:, 5:100], realspec[1][:, 100:]), axis=1)
    # mixedIF2 = np.concatenate((fakespec[1][:10, :], realspec[1][10:50, :], fakespec[1][50:, :]), axis=0)
    # mixed = np.stack((fakespec[0], realspec[1]), axis=0)
    # mixed2 = np.stack((realspec[0], fakespec[1]), axis=0)
    # mixed3 = np.stack((realspec[0], mixedIF1), axis=0)
    # mixed4 = np.stack((fakespec[0], realspec[1]), axis=0)
    # mixed5 = np.stack((realspec[0], ((fakespec[1] - fakeIFmean) / fakeIFstd) * realIFstd + realIFmean))
    # mixed6 = np.stack((realspec[0], np.zeros(realspec[1].shape) + realIFmean))
    # mixed7 = np.stack((realspec[0], ((eegspec[1] - eegIFmean) / eegIFstd) * realIFstd + realIFmean))
    # faketemp = IF_to_eeg(fakespec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # faketemp = mne.filter.filter_data(faketemp, 64, 0.5, None)
    # spec = librosa.stft(faketemp, n_fft=512, win_length=512, hop_length=8)
    # magnitude = np.log(np.abs(spec) + epsilon)[: 224]  # abs是求复数的模，log我觉得是因为原始值太小了
    # angle = np.angle(spec)[: 224]
    # filteredfake = torch.from_numpy(np.stack((magnitude, angle), axis=0)[None, :, :, :])
    # filteredfake = normalizer.normalize(filteredfake, type='seeg')[0].numpy()
    # realback = IF_to_eeg(realspec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # fakefilteredback = IF_to_eeg(filteredfake, normalizer, iseeg=False)[0]
    # mixedback = IF_to_eeg(mixed, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback = mne.filter.filter_data(mixedback, 64, 0.5, None)
    fakeback = IF_to_eeg(fakespec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback2 = IF_to_eeg(mixed2, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback3 = IF_to_eeg(mixed3, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback4 = IF_to_eeg(mixed4, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback4 = mne.filter.filter_data(mixedback4, 64, 0.5, None)
    # mixedback5 = IF_to_eeg(mixed5, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback6 = IF_to_eeg(mixed6, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback7 = IF_to_eeg(mixed7, normalizer, iseeg=False, is_IF=is_IF)[0]
    #
    # eegmixed = np.stack((eegspec[0], realspec[1]), axis=0)
    eegback = IF_to_eeg(eegspec[None, ...], normalizer, iseeg=True, is_IF=is_IF)[0]
    # fakeback_1 = IF_to_eeg(fakespec_1, normalizer, iseeg=False, is_IF=is_IF)[0]
    # IFback = IF_to_eeg(IFspec, normalizer, iseeg=False, is_IF=is_IF)[0]

    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_ylim([-0.0014, 0.0014])
    # ax[0].plot(np.arange(conf.audio_length), realback, label='real SEEG')
    # ax[0].plot(np.arange(conf.audio_length), fakeback, label='fake IF')
    # ax[0].legend()
    # ax[1].set_ylim([-0.0014, 0.0014])
    # ax[1].plot(np.arange(conf.audio_length), realback, label='real')
    # ax[1].plot(np.arange(conf.audio_length), fakeback, label='Ours')
    # ax[1].legend()
    # ax[2].set_ylim([-0.0014, 0.0014])
    # ax[2].plot(np.arange(conf.audio_length), realback, label='real')
    # ax[2].plot(np.arange(conf.audio_length), IFback, label='GAN IF')
    # ax[2].legend()
    # plt.savefig('/public/home/xlwang/hmq/tt1')
    # plt.plot()
    # plt.show()
    # psd_IF = mne.time_frequency.psd_array_welch(IFback, conf.seeg_sf, fmin=0, fmax=32, n_fft=conf.seeg_n_fft)[0]
    # psd_ours = mne.time_frequency.psd_array_welch(fakeback, conf.seeg_sf, fmin=0, fmax=32,
    #                                             n_fft=conf.seeg_n_fft)[0]
    # psd_real = mne.time_frequency.psd_array_welch(realback, conf.seeg_sf, fmin=0, fmax=32,
    #                                             n_fft=conf.seeg_n_fft)[0]
    # print(dtw.distance_fast(psd_IF, psd_real, use_pruning=True))
    # print(dtw.distance_fast(psd_os, psd_real, use_pruning=True))
    # info = mne.create_info(1, 64.)
    # raw1 = mne.io.RawArray(IFback[None, :], info)
    # raw2 = mne.io.RawArray(fakeback[None, :], info)
    # raw3 = mne.io.RawArray(realback[None, :], info)
    # raw1.plot_psd(ax=ax[0], picks='0', show=False)
    # raw2.plot_psd(ax=ax[1], picks='0', show=False)
    # raw3.plot_psd(ax=ax[2], picks='0', show=False)
    # plt.show()

    # realseeg = np.load("/home/hmq/Datasets/cv_0615_20/A/val/zxl/zxl_PZ_C1_1851.npy", allow_pickle=True)
    # realeeg = np.load("/home/hmq/Datasets/cv_0525/B/val/zxl/zxl_CZ_K11_1192.npy", allow_pickle=True)
    # fakeseegIF = np.load('/home/hmq/Projects/experiments/results/cv_pix2pix_resnet_0516_lk/val_latest/npys/zxl_O1_A1_1349_fake_B.npy', allow_pickle=True)[1]
    # seegspec = librosa.stft(realseeg[0], n_fft=512, win_length=512, hop_length=8)
    # seegmagnitude = np.log(np.abs(seegspec) + epsilon)[: 224]  # abs是求复数的模，log我觉得是因为原始值太小了
    # seegangle = np.angle(seegspec)[: 224]
    # seegIF = phase_operation.instantaneous_frequency(seegangle, time_axis=1)[: 224]
    # eegspec = librosa.stft(realeeg[0], n_fft=512, win_length=512, hop_length=8)
    # eegmagnitude = np.log(np.abs(eegspec) + epsilon)  # abs是求复数的模，log我觉得是因为原始值太小了
    # eegangle = np.angle(eegspec)[: 224]
    # eegIF = phase_operation.instantaneous_frequency(eegangle, time_axis=1)[: 224]
    # eegIFback = mag_plus_phase(eegmagnitude, eegIF, True, True)

    fake_s_spec = librosa.stft(eegback, n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                               hop_length=conf.seeg_hop)
    s_mag = np.log(np.abs(fake_s_spec) + conf.epsilon)[: conf.w]
    s_angle = np.angle(fake_s_spec)[: conf.w]
    s_IF = phase_operation.instantaneous_frequency(s_angle, time_axis=1)
    fake_s_spec = np.stack((s_mag, s_IF), axis=0)[None, ...]
    fake_s_spec = normalizer.normalize(torch.from_numpy(fake_s_spec), 'eeg').numpy()[0]
    fake_s_mag = fake_s_spec[0]
    fake_s_IF = fake_s_spec[1]
    # stacked = np.stack((F.softmax(torch.from_numpy(r_query[0].dot(r_key[0].transpose())), dim=-1).numpy().mean(0), F.softmax(torch.from_numpy(f_query[0].dot(f_key[0].transpose())), dim=-1).numpy().mean(0)), axis=0)
    fake_attned = F.softmax(torch.from_numpy(f_query[0].dot(f_key[0].transpose())), dim=-1).numpy()
    # plt.matshow(fake_attned)
    # plt.show()
    plot_spectrogram(np.arange(127), np.arange(128), save_path=None, vmin=None, vmax=None, show=True, r_key=fake_s_IF[:, 1:])
    # plot_spectrogram(np.arange(0, 224), np.arange(0, 100),
    #                  save_path=None, vmin=-1., vmax=1.,
    #                  realseeg=seegIF[:100, :], fakediff=fakeseegIF[:100, :], eegdiff=((fakeseegIF[:100, :] - np.mean(fakeseegIF[:100, :], axis=1)[:, None])/np.std(fakeseegIF[:100, :], axis=1)[:, None])
    #                 *np.std(seegIF[:100, :], axis=1)[:, None] + np.mean(seegIF[:100, :], axis=1)[:, None])
'''

'''
# 研究pghi和stft
data = np.random.rand(2048) * 10
    ap = AudioPreprocessor(sample_rate=128,
                           audio_length=2048,
                           transform='specgrams', hop_size=16,
                           stft_channels=256, win_size=256, fft_size=256, n_frames=128)  # , win_size=256, fft_size=256, n_frames=128
    preprocessor = ap.get_preprocessor()
    postprocessor = ap.get_postprocessor()
    print(data.shape)
    processed1 = preprocessor(data)
    processed2 = preprocessor(-data)
    postprocessed1 = postprocessor(processed1)
    postprocessed2 = postprocessor(processed2)
    spec1 = librosa.core.stft(
        data[: -1],
        hop_length=16,
        win_length=256,
        n_fft=256)
    spec2 = librosa.core.stft(
        -data[: -1],
        hop_length=8,
        win_length=256,
        n_fft=256)
    mag1 = np.log(np.abs(spec1))
    mag2 = np.log(np.abs(spec2))
    angle1 = np.angle(spec1)
    angle2 = np.angle(spec2)
    backmag = np.concatenate([np.exp(mag1[:-1]), np.exp(mag1[-1:])], axis=-2)
    angle1 = np.concatenate([angle1[:-1], angle1[-1:]], axis=-2)
    backspec = np.array(backmag) * np.exp(1.j * np.array(angle1))
    back1 = librosa.core.istft(
                    backspec,
                    hop_length=16,
                    win_length=256)
    # print('spec 1:', spec1[1:3][:2])
    # print('spec 2:', spec2[1:3][:2])
    # print('mag 1:', mag1[1:3][:2])
    # print('mag 2:', mag2[1:3][:2])
    # print('angle 1:', angle1[1:3][:2])
    # print('angle 2:', angle2[1:3][:2])
    # print("spec sum:", spec1+spec2)
    # print(data)
    # print(processed1.shape)
    # print(postprocessed1.shape)
    # print(postprocessed2.shape)
    print(back1.shape)
    plt.plot(np.arange(back1.shape[0]), data[ :back1.shape[0]], label='real')
    plt.plot(np.arange(back1.shape[0]), back1, label='x')
    plt.plot(np.arange(postprocessed1.shape[0]), postprocessed1[: postprocessed1.shape[0]], label='postprocessed1')
    # plt.plot(np.arange(postprocessed2.shape[0]), postprocessed2, label='-x')
    plt.legend()
    # plot_spectrogram(np.arange(0, 224), np.arange(0, 100),
                     #                  save_path=None, vmin=-1., vmax=1.,
                     #                  realeeg=eegIF[:100, :], normedeeg=eegIF[:100, :] - np.mean(
                     #         eegIF[:100, :], axis=1)[:, np.newaxis])
    plt.show()
'''

# 电极方位的影响
'''
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
'''


#  探索相位的影响
'''
# realspec = np.load('/home/hmq/Projects/experiments/results/cv_pix2pix_resnet_0516_lk/val_latest/npys/lmk_CZ_K4_1834_real_B.npy', allow_pickle=True)
    # fakespec = np.load('/home/hmq/Projects/experiments/results/cv_pix2pix_resnet_0516_lk/val_latest/npys/lmk_CZ_K4_1834_fake_B.npy', allow_pickle=True)
    # eegspec = np.load(
    #     '/home/hmq/Projects/experiments/results/cv_pix2pix_resnet_0516_lk/val_latest/npys/lmk_CZ_K4_1834_real_A.npy',
    #     allow_pickle=True)
    #
    # bound = 10
    # realIFmean = np.mean(realspec[1], axis=1)[:, None]
    # realIFstd = np.std(realspec[1], axis=1)[:, None]
    # fakeIFmean = np.mean(fakespec[1], axis=1)[:, None]
    # fakeIFstd = np.std(fakespec[1], axis=1)[:, None]
    # eegIFmean = np.mean(eegspec[1], axis=1)[:, None]
    # eegIFstd = np.std(eegspec[1], axis=1)[:, None]
    # print('mean shape:', realIFmean.shape)
    # print('std shape:', realIFstd.shape)
    # mixedIF1 = np.concatenate((realspec[1][:bound, :], fakespec[1][bound:, :]), axis=0)
    # mixedIF2 = np.concatenate((fakespec[1][:bound, :], realspec[1][bound:, :]), axis=0)
    # # mixedIF1 = np.concatenate((realspec[1][:10, :], fakespec[1][10:50, :], realspec[1][50:, :]), axis=0)
    # # mixedIF2 = np.concatenate((fakespec[1][:10, :], realspec[1][10:50, :], fakespec[1][50:, :]), axis=0)
    # mixed = np.stack((realspec[0], mixedIF1), axis=0)
    # mixed2 = np.stack((realspec[0], mixedIF2), axis=0)
    # mixed3 = np.stack((fakespec[0], mixedIF2), axis=0)
    # mixed4 = np.stack((realspec[0], fakespec[1]), axis=0)
    # mixed5 = np.stack((realspec[0], ((fakespec[1] - fakeIFmean) / fakeIFstd) * realIFstd + realIFmean))
    # mixed6 = np.stack((fakespec[0], ((fakespec[1] - fakeIFmean) / fakeIFstd) * realIFstd + realIFmean))
    # mixed7 = np.stack((realspec[0], ((eegspec[1] - eegIFmean) / eegIFstd) * realIFstd + realIFmean))
    # faketemp = IF_to_eeg(fakespec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # faketemp = mne.filter.filter_data(faketemp, 64, 0.5, None)
    # spec = librosa.stft(faketemp, n_fft=512, win_length=512, hop_length=8)
    # magnitude = np.log(np.abs(spec) + epsilon)[: 224]  # abs是求复数的模，log我觉得是因为原始值太小了
    # angle = np.angle(spec)[: 224]
    # filteredfake = torch.from_numpy(np.stack((magnitude, angle), axis=0)[None, :, :, :])
    # filteredfake = normalizer.normalize(filteredfake, type='seeg')[0].numpy()
    # realback = IF_to_eeg(realspec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # fakefilteredback = IF_to_eeg(filteredfake, normalizer, iseeg=False)[0]
    # mixedback = IF_to_eeg(mixed, normalizer, iseeg=False, is_IF=is_IF)[0]
    # fakeback = IF_to_eeg(fakespec, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback2 = IF_to_eeg(mixed2, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback3 = IF_to_eeg(mixed3, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback4 = IF_to_eeg(mixed4, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback5 = IF_to_eeg(mixed5, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback6 = IF_to_eeg(mixed6, normalizer, iseeg=False, is_IF=is_IF)[0]
    # mixedback7 = IF_to_eeg(mixed7, normalizer, iseeg=False, is_IF=is_IF)[0]
    #
    # eegmixed = np.stack((eegspec[0], realspec[1]), axis=0)
    # eegback = IF_to_eeg(eegmixed, normalizer, iseeg=False, is_IF=is_IF)[0]
    #
    # fig, ax = plt.subplots(3,1)
    # ax[0].plot(np.arange(1784), realback, label='real')
    # ax[0].plot(np.arange(1784), fakeback, label='eeg mag + seeg phase')
    # ax[0].legend()
    # ax[1].plot(np.arange(1784), realback, label='real')
    # ax[1].plot(np.arange(1784), mixedback5, label='real phase + fake mag')
    # ax[1].legend()
    # ax[2].plot(np.arange(1784), realback, label='real')
    # ax[2].plot(np.arange(1784), mixedback6, label='real mag + fake phase')
    # ax[2].legend()
    # plt.show()

    # realseeg = np.load("/home/hmq/Datasets/cv_0512/A/val/zxl/zxl_O1_A1_1349.npy", allow_pickle=True)
    # realeeg = np.load("/home/hmq/Datasets/cv_0512/B/val/zxl/zxl_O1_A1_1349.npy", allow_pickle=True)
    # fakeseegIF = np.load('/home/hmq/Projects/experiments/results/cv_pix2pix_resnet_0516_lk/val_latest/npys/zxl_O1_A1_1349_fake_B.npy', allow_pickle=True)[1]
    # seegspec = librosa.stft(realseeg[0], n_fft=512, win_length=512, hop_length=8)
    # seegmagnitude = np.log(np.abs(seegspec) + epsilon)[: 224]  # abs是求复数的模，log我觉得是因为原始值太小了
    # seegangle = np.angle(seegspec)[: 224]
    # seegIF = phase_operation.instantaneous_frequency(seegangle, time_axis=1)[: 224]
    # eegspec = librosa.stft(realeeg[0], n_fft=512, win_length=512, hop_length=8)
    # eegmagnitude = np.log(np.abs(eegspec) + epsilon)  # abs是求复数的模，log我觉得是因为原始值太小了
    # eegangle = np.angle(eegspec)[: 224]
    # eegIF = phase_operation.instantaneous_frequency(eegangle, time_axis=1)[: 224]
    # eegIFback = mag_plus_phase(eegmagnitude, eegIF, True, True)
    # plot_spectrogram(np.arange(0, 224), np.arange(0, 100), prefix + ' IF',
    #                  save_dir=None, vmin=-1., vmax=1.,
    #                  realdiff=seegIF[:100, :], fakediff=fakeseegIF[:100, :], eegdiff=((fakeseegIF[:100, :] - np.mean(fakeseegIF[:100, :], axis=1)[:, None])/np.std(fakeseegIF[:100, :], axis=1)[:, None])
    #                 *np.std(seegIF[:100, :], axis=1)[:, None] + np.mean(seegIF[:100, :], axis=1)[:, None])
    # for i, IF in enumerate([seegIF, fakeseegIF, eegIF]):
    #     mean = np.mean(IF, axis=1)
    #     std = np.std(IF, axis=1)
    #     print(i)
    #     print('mean:', mean)
    #     print('std:', std)
'''

# 生成训练集的样本索引
'''
    trainRands = {'lk':0, 'zxl': 0, 'lxh':0, 'lmk':0, 'tll':0, 'yjh':0, 'wzw':0}
    datasetName = 'cv_0615_100'
    maxDataset = 5000
    for p in trainRands.keys():
        lenDataset = len(make_dataset(os.path.join("/home/hmq/Datasets", datasetName, "A", "train", p)))
        if lenDataset > maxDataset:
            rands = random.sample(range(0, lenDataset), maxDataset)
            trainRands[p] = rands
        else:
            rands = random.sample(range(0, lenDataset), maxDataset - lenDataset)
            trainRands[p] = list(range(0, lenDataset)) + rands
        print(p)
        print(trainRands[p])
        print(len(trainRands[p]))
        print(min(trainRands[p]))
        print(max(trainRands[p]))
    np.save(os.path.join('/home/hmq/Datasets/', datasetName, 'dataIndex.npy'), trainRands)
'''

#  从train集中删除和val以及test存在片段交叉的数据
'''
    offset = [-3, -2, -1, 1, 2, 3]
    for p in all_patients:
        eegvaldir = os.path.join('/home/hmq/Datasets/cv_0512/B/val/', p)
        eegtestdir = os.path.join('/home/hmq/Datasets/cv_0512/B/test/', p)
        eegtraindir = os.path.join('/home/hmq/Datasets/cv_0512/B/train/', p)
        seegtraindir = os.path.join('/home/hmq/Datasets/cv_0512/A/train/', p)
        eegvaldataset = make_dataset(eegvaldir)
        for file in eegvaldataset:
            f_n = os.path.basename(file)
            stem = f_n.split('.')[0].split('_')[: -1]
            number = int(f_n.split('.')[0].split('_')[-1])
            for off in offset:
                newf_n = '_'.join(stem + [str(number + off)]) + '.npy'
                eegtrainpath = os.path.join(eegtraindir, newf_n)
                seegtrainpath = os.path.join(seegtraindir, newf_n)
                if os.path.exists(eegtrainpath):
                    os.remove(eegtrainpath)
                    print('remove {}'.format(eegtrainpath))
                    os.remove(seegtrainpath)
                    print('remove {}'.format(seegtrainpath))
        eegtestdataset = make_dataset(eegtestdir)
        for file in eegtestdataset:
            f_n = os.path.basename(file)
            stem = f_n.split('.')[0].split('_')[: -1]
            number = int(f_n.split('.')[0].split('_')[-1])
            for off in offset:
                newf_n = '_'.join(stem + [str(number + off)]) + '.npy'
                eegtrainpath = os.path.join(eegtraindir, newf_n)
                seegtrainpath = os.path.join(seegtraindir, newf_n)
                if os.path.exists(eegtrainpath):
                    os.remove(eegtrainpath)
                    print('remove {}'.format(eegtrainpath))
                    os.remove(seegtrainpath)
                    print('remove {}'.format(seegtrainpath))
'''


#  划分cv_0512数据集
'''
    pickedPairs = {}
    for p in all_patients:
        dist = np.load('/home/hmq/Infos/dist/' + p + '_pos_dist.npy', allow_pickle=True).item()
        pickedPairs[p] = []
        pickedSEEG = []
        for eeg in dist.keys():
            for pair in dist[eeg]:
                if 100 < pair[1] and pair[1] < 110 and pair[0] not in pickedSEEG:
                    pickedPairs[p].append([eeg, *pair])
                    pickedSEEG.append(pair[0])
                    break
        print(p)
        print(pickedPairs[p])
    np.save('/home/hmq/Datasets/cv_0512/pickedPairs', pickedPairs)
    pickedPairs = np.load('/home/hmq/Datasets/cv_0512/pickedPairs.npy', allow_pickle=True).item()
    print([(k, len(v)) for k,v in pickedPairs.items()])
    for p, pairs in pickedPairs.items():
        eeg_save_dir = os.path.join("/home/hmq/Datasets", datasetName, "B", "train", p)
        seeg_save_dir = os.path.join("/home/hmq/Datasets", datasetName, "A", "train", p)
        all_eeg = [pair[0] for pair in pairs]
        all_seeg = [pair[1] for pair in pairs]
        raw_seeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_seeg_raw.fif'))
        raw_eeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_eeg_raw.fif'))
        raw_seeg.pick_channels(all_seeg)
        raw_eeg.pick_channels(all_eeg)

        if p == 'lmk':
            annot = mne.read_annotations("/home/hmq/Infos/annotations/lmk_bad_spans.csv")
            bad_intvs = [[math.floor(t['onset']), math.ceil(t['onset'] + t['duration'])] for t in annot[4:]]

        for pair in pairs:
            pickedseeg = raw_seeg.copy().pick_channels([pair[1]])
            pickedeeg = raw_eeg.copy().pick_channels([pair[0]])
            if p != 'lmk':
                slice_data(pickedeeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=0,
                           prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
                slice_data(pickedseeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=0,
                           prefix=p + '_' + pair[0] + '_' + pair[1] + '_')

            else:
                start_number = 0
                start = 0
                end = bad_intvs[0][0]
                for idx in range(len(bad_intvs)):

                    cropped_eeg = pickedeeg.copy().crop(tmin=start, tmax=end)
                    cropped_seeg = pickedseeg.copy().crop(tmin=start, tmax=end)

                    slice_data(cropped_eeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                               prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
                    next = slice_data(cropped_seeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                                      prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
                    del cropped_eeg
                    del cropped_seeg

                    if idx == len(bad_intvs) - 1:
                        break
                    start = bad_intvs[idx][1]
                    end = bad_intvs[idx + 1][0]
                    start_number = next

    for p in all_patients:
        seegsrc = os.path.join('/home/hmq/Datasets/cv_0512/A/train', p)
        eegsrc = os.path.join('/home/hmq/Datasets/cv_0512/B/train', p)
        seegdataset = make_dataset(seegsrc)
        eegdataset = make_dataset(eegsrc)
        rd = random.sample(range(0, len(seegdataset)), 100)
        t_no = rd[: 50]
        v_np = rd[50:]
        for i, f in enumerate(seegdataset):
            f_n = os.path.basename(f)
            if i in t_no:
                shutil.move(os.path.join(eegsrc, f_n), os.path.join('/home/hmq/Datasets/cv_0512/B/test', p, f_n))
                shutil.move(os.path.join(seegsrc, f_n), os.path.join('/home/hmq/Datasets/cv_0512/A/test', p, f_n))
            elif i in v_np:
                shutil.move(os.path.join(eegsrc, f_n), os.path.join('/home/hmq/Datasets/cv_0512/B/val', p, f_n))
                shutil.move(os.path.join(seegsrc, f_n), os.path.join('/home/hmq/Datasets/cv_0512/A/val', p, f_n))
        print('eeg train size:', len(make_dataset(eegsrc)))
        print('seeg train size:', len(make_dataset(seegsrc)))
        print('eeg test size:', len(make_dataset(os.path.join('/home/hmq/Datasets/cv_0512/B/test', p))))
        print('seeg test size:', len(make_dataset(os.path.join('/home/hmq/Datasets/cv_0512/A/test', p))))
        print('eeg val size:', len(make_dataset(os.path.join('/home/hmq/Datasets/cv_0512/B/val', p))))
        print('seeg val size:', len(make_dataset(os.path.join('/home/hmq/Datasets/cv_0512/A/val', p))))
'''


#  计算EEG和SEEG的距离与相似度的关系
'''
    for p in all_patients:
        all = explore_segment_dist_sim_relation(p, "/home/hmq/Infos/dist/" + p + "_pos_dist.npy", saveDir="/home/hmq/Infos/distSimCorr/")
        np.save('/home/hmq/Infos/distSimCorr/' + p + '_segmentResult', all)
'''



#  划分cv_0415数据集
'''
    for p, pair in eeg_seeg_pairs.items():
        eeg_save_dir = os.path.join("/home/hmq/Datasets", dataset_name, "B", "train", p)
        seeg_save_dir = os.path.join("/home/hmq/Datasets", dataset_name, "A", "train", p)
        if p != 'lmk':
            continue
            raw_seeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_seeg_raw.fif'))
            raw_seeg.pick_channels([pair[1]])
            raw_eeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_eeg_raw.fif'))
            raw_eeg.pick_channels([pair[0]])
            slice_data(raw_eeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=0, prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
            slice_data(raw_seeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=0,
                       prefix=p + '_' + pair[0] + '_' + pair[1] + '_')

        else:
            raw_seeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_seeg_raw.fif'))
            raw_seeg.pick_channels([pair[1]])
            raw_eeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_eeg_raw.fif'))
            raw_eeg.pick_channels([pair[0]])
            annot = mne.read_annotations("/home/hmq/Infos/annotations/lmk_bad_spans.csv")
            bad_intvs = [[math.floor(t['onset']), math.ceil(t['onset'] + t['duration'])] for t in annot[4:]]
            start_number = 0
            start = 0
            end = bad_intvs[0][0]
            for idx in range(len(bad_intvs)):

                cropped_eeg = raw_eeg.copy().crop(tmin=start, tmax=end)
                cropped_seeg = raw_seeg.copy().crop(tmin=start, tmax=end)

                slice_data(cropped_eeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                           prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
                next = slice_data(cropped_seeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                           prefix=p + '_' + pair[0] + '_' + pair[1] + '_')
                del cropped_eeg
                del cropped_seeg

                if idx == len(bad_intvs) - 1:
                    break
                start = bad_intvs[idx][1]
                end = bad_intvs[idx + 1][0]
                start_number = next
'''


# nearset = {'lk': 'F11', 'zxl': 'C1', 'yjh': 'G10', 'tll': 'E2', 'lmk': 'H14', 'lxh': 'L17', 'wzw': 'K2'}

# dist = {}
# ls = []
# for p in nearset.keys():
#     dist[p] = {}
#     sim_info = np.load(os.path.join("/home/hmq/Infos/dist/", p + '_pos_dist.npy'), allow_pickle=True).item()
#     for k, v in sim_info.items():
#         for t in v:
#             if t[0] == nearset[p]:
#                 dist[p][k] = t[1]
#     ls.append(sorted(dist[p].items(), key=lambda x: x[1]))
# np.save("/home/hmq/Infos/ch_names/seeg_ch_for_corr.npy", [nearset, ls])


# fake_spec = []
# for j in range(fake_s_temporal.shape[0]):
#     e_spec = librosa.stft(fake_s_temporal[j], n_fft=conf.eeg_n_fft, win_length=conf.eeg_win_len,
#                           hop_length=conf.eeg_hop)
#     e_spec = np.log(np.abs(e_spec) + conf.epsilon)[: conf.w]
#     e_spec = np.stack((e_spec, np.zeros(e_spec.shape)), axis=0)
#     fake_spec.append(e_spec)
# fake_spec = np.asarray(fake_spec)
# fake_spec = normalizer.normalize(torch.from_numpy(fake_spec), 'eeg').numpy()[0][:1, ...]
# plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
#                  ' Spectrogram clipped', vmin=-0.9, vmax=0.9, save_dir=None,
#                  Fake=fake[0], Transferred=fake_spec[0])

"""
#  下面是计算距离
for p in all_patients:
    results = calculate_distance('/home/hmq/Datasets/cv/A/val',
                                 # os.path.join("/home/hmq/Projects/experiments/results/", exper_name, "val_latest", p),
                                 os.path.join('/home/hmq/Datasets/cv/B/val', p),
                                 normalizer, method='EEG', save_path=os.path.join(stat_dir, exper_name, "_".join([p, "eeg", "val", "psd_dist"])))
    print("Patient {}:".format(p))
    print(results)
"""

'''
# 下面是生成val数据集的代码
patients = ['lk']
n_file = 10315
n_sample = 50
rd = list(range(1, n_file-1, 1 + (n_file-2) // n_sample))
print(rd)
print(len(rd))
for patient in patients:
    eeg_dataset = make_dataset(os.path.join('/home/hmq/Datasets/cv/B/train', patient))
    seeg_dataset = make_dataset(os.path.join('/home/hmq/Datasets/cv/A/train', patient))
    for i in range(len(eeg_dataset)):
        if i in rd:
            shutil.move(eeg_dataset[i], os.path.join('/home/hmq/Datasets/cv/B/val', patient))
            shutil.move(seeg_dataset[i], os.path.join('/home/hmq/Datasets/cv/A/val', patient))
'''

# 下面是分割数据(cv_0407)的代码
# tll pairs [('FZ', 'I8', array([69.34300274])), ('F4', 'A8', array([73.33140372])), ('PZ', 'E2', array([50.0013947]))]
# lk pairs [('Fp2', 'E10', array([20.69592009])), ('FZ', 'F1', array([24.33117395])), ('F4', 'F11', array([9.40280172])), ('F8', 'B14', array([14.80064537])), ('C4', 'D15', array([34.86986275])), ('T4', 'H14', array([19.30499903]))]
# zxl pairs [('CZ', 'D1', array([28.5934427])), ('PZ', 'C1', array([20.97461499])), ('O1', 'L1', array([46.05459558]))]
# yjh pairs [('F3', 'E10', array([44.71298704])), ('CZ', 'L3', array([69.89194194])), ('C3', 'G10', array([39.84213185]))]
# lmk pairs [('CZ', 'A14', array([9.39597045])), ('C3', 'B10', array([28.10216665])), ('T3', 'F3', array([38.61271169])), ('P4', 'H14', array([14.89213845]))]
# lxh pairs [('CZ', 'L17', array([20.55346079])), ('C3', 'H8', array([8.27186601])), ('T3', 'H3', array([56.06507585])), ('PZ', 'G10', array([21.1602206]))]
# wzw pairs [('PZ', 'G4', array([55.29771476])), ('O2', 'K2', array([41.76180341]))]
# eeg_seeg_pairs = {'tll': ['PZ', 'E2'], 'lk': ['F4', 'F11'], 'zxl': ['PZ', 'C1'], 'yjh': ['C3', 'G10'],
#                   'lxh': ['C3', 'H8'], 'wzw': ['O2', 'K2'], 'lmk': ['CZ', 'A14']}
'''
nearset, eeg_dist = np.load("/home/hmq/Infos/ch_names/seeg_ch_for_corr.npy", allow_pickle=True)
# for i, p in enumerate(list(nearset.keys())[1:]):
p = 'lmk'
i = list(nearset.keys()).index(p)
raw_seeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_seeg_raw.fif'))  # Sleep_Aug_4th_2am
raw_seeg.pick_channels([nearset[p]])
raw_eeg = read_raw_signal(os.path.join("/home/hmq/Signal/preprocessed/", p + '_eeg_raw.fif'))
annot = mne.read_annotations("/home/hmq/Infos/annotations/lmk_bad_spans.csv")
bad_intvs = [[math.floor(t['onset']), math.ceil(t['onset'] + t['duration'])] for t in annot[4:]]

for pair in eeg_dist[i]:

    picked_eeg = raw_eeg.copy().pick_channels([pair[0]])
    start_number = 0
    start = 0
    end = bad_intvs[0][0]

    if pair[1] < 50:
        eeg_save_dir = "/home/hmq/Datasets/cv_0407/corr/B/near/" + p
        seeg_save_dir = "/home/hmq/Datasets/cv_0407/corr/A/near/" + p
    elif pair[1] > 110:
        eeg_save_dir = "/home/hmq/Datasets/cv_0407/corr/B/far/" + p
        seeg_save_dir = "/home/hmq/Datasets/cv_0407/corr/A/far/" + p
    else:
        eeg_save_dir = "/home/hmq/Datasets/cv_0407/B/train/" + p
        seeg_save_dir = "/home/hmq/Datasets/cv_0407/A/train/" + p

    for idx in range(len(bad_intvs)):

        cropped_eeg = picked_eeg.copy().crop(tmin=start, tmax=end)
        cropped_seeg = raw_seeg.copy().crop(tmin=start, tmax=end)

        slice_data(cropped_eeg, eeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                   prefix=p + '_' + pair[0] + '_' + nearset[p] + '_')
        next = slice_data(cropped_seeg, seeg_save_dir, 1784, hop=1784 // 4, start_number=start_number,
                   prefix=p + '_' + pair[0] + '_' + nearset[p] + '_')
        del cropped_eeg
        del cropped_seeg

        if idx == len(bad_intvs) - 1:
            break
        start = bad_intvs[idx][1]
        end = bad_intvs[idx + 1][0]
        start_number = next
    del picked_eeg
'''

# eeg = ['EEG FZ-Ref']
# seeg = ['EEG F1-Ref']
# ref = ['POL B6']
# interpolation = ['EEG FZ-Ref', 'EEG F4-Ref-1', 'EEG F3-Ref-1', 'EEG CZ-Ref', 'EEG Fp2-Ref']  # [ 8.19437855 41.31195912 44.70376615 56.83378265 63.77621662]
# nearest_seeg = ['EEG F2-Ref', 'EEG F3-Ref-0', 'EEG F4-Ref-0', 'EEG F5-Ref', 'EEG A2-Ref-0', 'EEG A1-Ref-0',
#                 'POL A3', 'EEG F6-Ref', 'POL A4', 'POL A5']
# itv = [(3600, 4320), (10800, 11520)]
# sf = 64
# eeg_pos = np.load("/home/hmq/Infos/position_info/eeg_pos.npy", allow_pickle=True).item()
# picked_pos = {}
# picked_pos['EEG F1-Ref'] = [4.04258916, 35.95939224, 48.34240927]
# for k, v in eeg_pos.items():
#     for ch in interpolation:
#         if k.lower() in ch.lower():
#             picked_pos[ch] = list(v)
# eeg_ch = np.load('/home/hmq/Infos/position_info/eeg_pos.npy', allow_pickle=True).item()

'''
# 下面两行是划分cv数据集的
# pairs = [('EEG F3-Ref-1', 'POL E10'), ('EEG CZ-Ref', 'POL L3'), ('EEG C3-Ref-1', 'POL G10')]
# slice_eeg_seeg(pairs, "/home/hmq/Signal/raw/Yin_JH.edf", '/home/hmq/Datasets/four/B/yjh/', '/home/hmq/Datasets/four/A/yjh/', 'yjh')
'''

# raw.crop(tmin=11520, tmax=None)  # 只要EEG和SEEG都统一操作，不用关心结束/开始的这一秒有没有重复！！
# raw.notch_filter(np.arange(50., int(sf / 2 / 50) * 50 + 1., 50), fir_design='firwin')
# for ch in interpolation:
#     path = os.path.join("/home/hmq/Datasets/LK_rest_Fz_224x224_e2s_28Hz_middle/", ch)
#     if not os.path.exists(path):
#         os.mkdir(path)
# for ch in nearest_seeg:
#     path = os.path.join("/home/hmq/Datasets/LK_rest_Fz_224x224_e2s_28Hz_middle/", ch)
#     raw = read_raw_signal("/home/hmq/Signals/huashan_rawdata_1/LK_Sleep_Aug_4th_2am.edf")  # Sleep_Aug_4th_2am
#     raw.pick_channels([ch])
#     raw.resample(sf, npad="auto")
#     raw.filter(.5, None, fir_design='firwin')
#     raw1 = raw.copy().crop(tmin=3600, tmax=4320)
#     slice_data(raw1, path, 1784, hop=1784 // 4, start_number=513)
#     raw.crop(tmin=10800, tmax=11520)
#     slice_data(raw, path, 1784, hop=1784 // 4, start_number=1539)
#     del raw
#     del raw1
# copy_random_files(1887, 376, src="/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/", dest="/home/cbd109/Users/Data/hmq/GANDatasets/LK_rest_Fz_64x64/")
# "/home/cbd109/Users/hmq/huashan_data_processed/LK/LK_rest/LK_Sleep_Aug_4th_2am_eeg_raw.fif"

# data = raw[:, :]
# plt.plot(data[1], data[0].T)
# plt.show()

'''normalizer = DataNormalizer(None)
scores = metric.compute_score_raw(1, "/home/cbd109-3/Users/data/hmq/GANDatasets/LK_rest/B/test/",
                                  "/home/cbd109-3/Users/data/hmq/codes/pix2pix/results/AE_IF/test_latest/npys/", None,
                                  normalizer=normalizer)
metric.print_score(scores)'''
# plot_tsne('real_B', 'fake_B', "/home/cbd109/Users/data/hmq/codes/pix2pix/results/AE_IF/test_latest/npys/", '3d', azim=0, elev=90)

'''这里是计算训练集的max_min归一化参数
real_train_path = '/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/train/'
real_test_path = '/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/test/'
fake_path = '/home/cbd109/Users/hmq/codes/pix2pix/results/IF_GAN/test_latest/npys/'
normalizer = DataNormalizer(None)
dataset = metric.EEG(real_train_path, isreal=True, transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
#dataset = EEG(fake_path, isreal=False, isIF=True, normalizer=normalizer,
#              transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
metric.get_max_min_normalize_params(dataloader, 'IF_GAN_maxmin.npy')'''

# results = np.load('/home/hmq/Infos/evaluation/stft_Fz_224x224_ae_e2s_dist_spline_with_hamming.npy', allow_pickle=True).item()
# print(results)
# bins = [0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58]
# plot_hist_distribution(results['mag_array'][0], bins, 'Distance', 'Density', '0 to 4Hz Magnitude Distance Distribution',
#                        save_dir='/home/hmq/Projects/pix2pix/')
# plot_hist_distribution(results['mag_array'][1], bins, 'Distance', 'Density', '4 to 7Hz Magnitude Distance Distribution',
#                        save_dir='/home/hmq/Projects/pix2pix/')
# plot_hist_distribution(results['mag_array'][2], bins, 'Distance', 'Density', '8 to 15Hz Magnitude Distance Distribution',
#                        save_dir='/home/hmq/Projects/pix2pix/')
# plot_hist_distribution(results['mag_array'][3], bins, 'Distance', 'Density', '16 to 28Hz Magnitude Distance Distribution',
#                        save_dir='/home/hmq/Projects/pix2pix/')
# print(results['temporal_mean'])
# print(results['temporal_std'])
# plot_mean_std(['0-4Hz', '4-7Hz', '8-15Hz', '16-28Hz'], results['mag_mean'], results['mag_std'],
#               title='Magnitude Mean and Std', save_dir='/home/hmq/Projects/pix2pix/')

# 'temporal_mean': temporal_mean, 'temporal_std': temporal_std, 'mag_mean': mag_mean, 'mag_std': mag_std,
# 'shrunk_mag_dist': shrunk_mag_dist, 'phase_mean': phase_mean, 'phase_std': phase_std,
# 'temporal_array': temporal_array, 'mag_array': mag_array
# for f_n in dataset[:30]:
#     f_n = os.path.basename(f_n)
#     n = f_n.split('.')[0].split('_')[0]
#     real_e = np.load(os.path.join("/home/hmq/Datasets/LK_rest_Fz_224x224_e2s_28Hz/B/test/", str(n) + '.npy'))
#     real_e = librosa.stft(real_e[0], n_fft=512, win_length=512, hop_length=8)
#     real_e = np.log(np.abs(real_e) + 1.0e-6)[:224]
#     real_e = normalizer.normalize(torch.from_numpy(real_e), 'eeg').numpy()[0][0]
#     real_s = np.load(
#         os.path.join("/home/hmq/Datasets/LK_rest_Fz_224x224_e2s_28Hz_middle/A/test",
#                      str(n) + '.npy'))
# real_s = IF_to_eeg(real_s, normalizer, iseeg=False)
# mne.filter.filter_data(real_s,sfreq=64, h_freq=28)
# fake_s = np.load(os.path.join('/home/hmq/Projects/pix2pix/results/stft_Fz_224x224_ae_e2s_28Hz_middle/test_latest/npys/',
#                               str(n) + '_fake_B.npy'))
# fake_s = IF_to_eeg(fake_s, normalizer, iseeg=False)
# draw_comparable_figure('Time(s)', 'Magnitude', seeg_sf, s_start=0, s_end=None, ch_intv=0, show=False,
#                        save_path="/home/hmq/Projects/pix2pix/results/stft_Fz_224x224_ae_e2s_28Hz_D_ahead/test_latest/temporal/" + str(n) + '.png',
#                        Real_SEEG=real_s[np.newaxis, :], Fake_SEEG=fake_s[np.newaxis, :])
# real_s = np.load(os.path.join('/home/hmq/Projects/pix2pix/results/ASAE_Fz_e2s_28Hz_from_scratch_middle/test/npys/', str(n) + '_real_seeg.npy'))[0]
# fake_s = np.load(os.path.join('/home/hmq/Projects/pix2pix/results/ASAE_Fz_e2s_28Hz_from_scratch_middle/test/npys/',
#                               str(n) + '_fake_seeg.npy'))[0]
# plot_spectrogram(np.arange(0, 224), np.arange(0, 28, .125), real_s, title='Real SEEG ' + str(n),
#                  save_dir='/home/hmq/Projects/pix2pix/results/ASAE_Fz_e2s_28Hz_from_scratch_middle/test/npys_spectro/', vmin=-.9,vmax=.9)
# plot_spectrogram(np.arange(0, 224), np.arange(0, 28, .125), fake_s, title='Fake SEEG ' + str(n),
#                  save_dir='/home/hmq/Projects/pix2pix/results/ASAE_Fz_e2s_28Hz_from_scratch_middle/test/npys_spectro/',vmin=-.9,vmax=.9)
# np.save('/home/cbd109-3/Users/data/hmq/norm_args/fake_real_band_distance.npy', results)
# n_fft = 512
# win_length = 512
# hop = 32
# fake_B = fake_B.astype(np.float64)
# b, a = scipy.signal.butter(5, [0.03125, 0.0546875], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
# real_B = scipy.signal.filtfilt(b, a, real_B[0])
# real_B = real_B.astype(np.float64)
# real_B = mne.filter.filter_data(real_B, 256., 20, 50)[0]
# fake_B = mne.filter.filter_data(np.expand_dims(fake_B, axis=0), 256., 20, 50)[0]
# spec = librosa.stft(fake_B, n_fft=n_fft, win_length=win_length, hop_length=hop)
# #magnitude = np.log(np.abs(spec) + 1.0e-8)[:256]
# magnitude=(np.abs(spec) + 1.0e-16)[: 256].T
# phase = np.angle(spec)[: 256].T
# phase = np.where(abs(phase) < 3.14, 0, np.sign(phase) * 1)
# print(phase[0])
# #psds, freqs = mne.time_frequency.psd_array_multitaper(real_B, sfreq=256)
# #print(psds)
# #print(spec.T[0])
# plt.plot(phase[0])
# plt.show()
#
# r_spec = librosa.stft(real_B, n_fft=n_fft, win_length=win_length, hop_length=hop)
# #r_magnitude = np.log(np.abs(r_spec) + 1.0e-8)[: 8]
# r_magnitude = np.abs(r_spec)[: 8]
# r_phase = np.angle(r_spec)[: 224]
# f_spec = librosa.stft(fake_B, n_fft=n_fft, win_length=win_length, hop_length=hop)
# #f_magnitude = np.log(np.abs(f_spec) + 1.0e-8)[: 8]
# f_magnitude = np.abs(f_spec)[: 8]
# f_phase = np.angle(f_spec)[: 224]
# print('Real Magnitude:\n', r_magnitude)
# print('Fake Magnitude:\n', f_magnitude)
# print('Difference:\n', np.linalg.norm(r_magnitude - f_magnitude))

'''接下来是比较groundtruth的代码'''
'''
dist_result = {}
dist_result['gan'] = np.load("/home/hmq/Infos/evaluation/stft_Fz_224x224_ae_e2s_dist_middle_with_hamming.npy", allow_pickle=True).item()
dist_result['asae'] = np.load("/home/hmq/Infos/evaluation/stft_Fz_224x224_asae_dist_middle_with_hamming.npy", allow_pickle=True).item()
dist_result['EUD'] = np.load("/home/hmq/Infos/evaluation/EUD_dist.npy", allow_pickle=True).item()
dist_result['spline'] = np.load("/home/hmq/Infos/evaluation/spline_dist.npy", allow_pickle=True).item()
dist_result['A1'] = np.load("/home/hmq/Infos/evaluation/EEG A1-Ref-0_dist.npy", allow_pickle=True).item()
dist_result['A2'] = np.load("/home/hmq/Infos/evaluation/EEG A2-Ref-0_dist.npy", allow_pickle=True).item()
dist_result['A3'] = np.load("/home/hmq/Infos/evaluation/POL A3_dist.npy", allow_pickle=True).item()
dist_result['A4'] = np.load("/home/hmq/Infos/evaluation/POL A4_dist.npy", allow_pickle=True).item()
dist_result['A5'] = np.load("/home/hmq/Infos/evaluation/POL A5_dist.npy", allow_pickle=True).item()
dist_result['F2'] = np.load("/home/hmq/Infos/evaluation/EEG F2-Ref_dist.npy", allow_pickle=True).item()
dist_result['F3'] = np.load("/home/hmq/Infos/evaluation/EEG F3-Ref-0_dist.npy", allow_pickle=True).item()
dist_result['F4'] = np.load("/home/hmq/Infos/evaluation/EEG F4-Ref-0_dist.npy", allow_pickle=True).item()
dist_result['F5'] = np.load("/home/hmq/Infos/evaluation/EEG F5-Ref_dist.npy", allow_pickle=True).item()
dist_result['F6'] = np.load("/home/hmq/Infos/evaluation/EEG F6-Ref_dist.npy", allow_pickle=True).item()
temp = {}
mag_1 = {}
mag_2 = {}
mag_3 = {}
mag_4 = {}
for k, v in dist_result.items():
    temp[k] = v['temporal_mean']
    mag_1[k] = v['mag_mean'][0][0]
    mag_2[k] = v['mag_mean'][1][0]
    mag_3[k] = v['mag_mean'][2][0]
    mag_4[k] = v['mag_mean'][3][0]
temp = sorted(temp.items(), key=lambda x: x[1])
mag_1 = sorted(mag_1.items(), key=lambda x: x[1])
mag_2 = sorted(mag_2.items(), key=lambda x: x[1])
mag_3 = sorted(mag_3.items(), key=lambda x: x[1])
mag_4 = sorted(mag_4.items(), key=lambda x: x[1])
print(temp)
print(mag_1)
print(mag_2)
print(mag_3)
print(mag_4)
'''

# 下面是求距离最大最小的图
'''dataset = make_dataset('/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/val/')
normalizer = DataNormalizer(None)
top_n = 20
dist = []
f_ns = []
style = 'L2'
for f_n in dataset:
    f_n = os.path.basename(f_n)
    n = f_n.split('.')[0]
    real = np.load(os.path.join('/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/val/', f_n))
    fake = np.load(os.path.join('/home/cbd109/Users/hmq/codes/pix2pix/results/IF_GAN/val_latest/npys/', n+'_fake_B.npy'))
    fake = IF_to_eeg(fake, normalizer)
    dist.append(calculate_distance(real, fake, style))
    f_ns.append(n)

f = [t[1] for t in sorted(zip(dist, f_ns))][:top_n]
print(f)'''