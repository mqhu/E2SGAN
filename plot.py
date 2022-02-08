import torch
from util.eeg_tools import *
import matplotlib.pyplot as plt
from gansynth.normalizer import DataNormalizer
from util.numpy_tools import plot_spectrogram
# from PGHI.preprocessing import AudioPreprocessor


def plot_phase_test(real_phase, fake_phase, real_temp, normalizer, conf, temporal_save_path, mag_save_path):
    real_s_spec = []
    for j in range(real_temp.shape[0]):
        s_spec = librosa.stft(real_temp[j], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                              hop_length=conf.seeg_hop)
        s_mag = np.log(np.abs(s_spec) + conf.epsilon)[: conf.w]
        s_angle = np.angle(s_spec)
        s_IF = phase_operation.instantaneous_frequency(s_angle, time_axis=1)[: conf.w]
        s_spec = np.stack((s_mag, s_IF), axis=0)
        real_s_spec.append(s_spec)
    real_s_spec = normalizer.normalize(torch.from_numpy(np.asarray(real_s_spec)), 'seeg').numpy()[0]

    mixed = np.stack((real_s_spec[0], fake_phase), axis=0)
    mixedback = IF_to_eeg(mixed, normalizer, iseeg=False, is_IF=is_IF)[0]

    plt.plot(np.arange(1016), real_temp[0], label='real')
    plt.plot(np.arange(1016), mixedback, label='real mag + fake phase')
    plt.legend()
    plt.savefig(temporal_save_path)
    plt.close()
    draw_comparable_figure('Time(1/64 s)', 'Magnitude', conf.seeg_sf, s_start=0, s_end=None, ch_intv=0, show=False,
                                                  save_path=os.path.join(temporal_save_dir, f_n + '_align' + '.png'),
                                                   Real_SEEG=real_temp, Fake_SEEG=mixedback[None, :])
    plot_spectrogram(np.arange(0, 128), np.arange(0, 128), save_path=mag_save_path, vmin=-1., vmax=1.,
                     realIF=real_phase, fakeIF=fake_phase, diff=real_phase-fake_phase)


if __name__ == '__main__':

    leave_out_patient = 'lxh'
    home_dir = '/public/home/xlwang'
    # home_dir = '/home'
    result_dir = os.path.join(home_dir, 'hmq/Projects/experiments/results')
    exper_name = "cv_pix2pix_global_onesided_0908"
    exper_dir = os.path.join(result_dir, exper_name + '_' + leave_out_patient, 'test_16')
    dataset_name = 'cv_0704_60'
    src_dir = os.path.join(home_dir, 'hmq/Datasets/' + dataset_name)
    # normalizer_dir = '/home/hmq/Infos/norm_args/'
    normalizer_dir = os.path.join(home_dir, 'hmq/Infos/norm_args/')

    bestK = np.load("/public/home/xlwang/hmq/Infos/bestK/best20.npy", allow_pickle=True).item()
    metric = 'bd'  # ['temporal_dist', 'rmse_dist', 'psd_dist', 'hd', 'bd']

    dataset = make_dataset(os.path.join(exper_dir, 'npys'))
    # dataset = bestK[metric]
    temporal_save_dir = os.path.join(exper_dir, 'temporal')
    mag_save_dir = os.path.join(exper_dir, 'mag')
    # temporal_save_dir = os.path.join('/public/home/xlwang/hmq/Projects/experiments/results/best20/', metric, 'temporal')
    # mag_save_dir = os.path.join('/public/home/xlwang/hmq/Projects/experiments/results/best20/', metric, 'mag')
    phase = 'test'  # test和train都是train目录
    is_IF = True
    is_ae = False
    is_pghi = False
    is_temporal = False
    is_asae = False
    is_phase = False
    domain = 'temporal' if is_temporal else 'freq'

    if not os.path.exists(temporal_save_dir):
        os.mkdir(temporal_save_dir)
    if not os.path.exists(mag_save_dir):
        os.mkdir(mag_save_dir)

    conf = Configuration()

    if is_pghi:
        ap = AudioPreprocessor(sample_rate=conf.seeg_sf,
                                          audio_length=conf.audio_length,
                                          transform='pghi', hop_size=conf.seeg_hop,
                                          stft_channels=conf.seeg_n_fft )
        preprocessor = ap.get_preprocessor()
        postprocessor = ap.get_post_processor()
    # for i in range(0, len(dataset), 3):
    #     if i > 50:
    #         break
    for i in range(len(dataset)):

        print('Plotting picture {}...'.format(i))
        f_n = os.path.basename(dataset[i])
        f_n = f_n.split('.')[0]
        patient = f_n.split("_")[0]
        f_n = '_'.join(f_n.split("_")[: 4])
        # exper_dir = '/'.join(dataset[i].split('/')[:-2])
        normalizer_name = dataset_name + '_without_' + patient

        if is_IF:
            normalizer_name += '_IF'
        if is_pghi:
            normalizer_name += '_pghi'
        if is_temporal:
            normalizer_name += '_temporal'
        normalizer_name += '.npy'
        normalizer = DataNormalizer(None, os.path.join(normalizer_dir, normalizer_name), False, domain=domain,
                                    use_phase=not is_pghi)

        real_e_temporal = np.load(os.path.join(src_dir, 'B', phase, patient, f_n + '.npy'))[:, : conf.audio_length]
        real_s_temporal = np.load(os.path.join(src_dir, 'A', phase, patient, f_n + '.npy'))[:, : conf.audio_length]

        if not is_pghi and conf.seeg_ceiling_freq < conf.seeg_sf // 2:
            real_e_temporal = mne.filter.filter_data(real_e_temporal, sfreq=conf.seeg_sf, l_freq=0, h_freq=conf.seeg_ceiling_freq)
            real_s_temporal = mne.filter.filter_data(real_s_temporal, l_freq=0, sfreq=conf.seeg_sf, h_freq=conf.seeg_ceiling_freq)
        if is_pghi:
            real_e_mag = preprocessor(real_e_temporal[0])[:, : conf.h, : conf.w]
            real_e_mag = normalizer.normalize(torch.from_numpy(np.asarray(real_e_mag[np.newaxis, :])), 'eeg').numpy()[0][0]
            real_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_real_B.npy'))
            fake_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_B.npy'))
            real_s_mag = real_s_spec[0]
            fake_s_temporal = pghi_invert(fake_s_spec[0], normalizer, postprocessor, False)
            fake_s_mag = preprocessor(fake_s_temporal[0])[:, : conf.h, : conf.w]
            fake_s_mag = normalizer.normalize(torch.from_numpy(np.asarray(fake_s_mag[np.newaxis, :])), 'seeg').numpy()[0][0]
        # elif 'pix2pix' in exper_name or 'gan' in exper_name or is_ae:
        elif not is_temporal:
            # real_s_spec = np.load(os.path.join(result_dir, leave_out_patient, f_n + '_real_B.npy'))
            # fake_s_spec = np.load(os.path.join(result_dir, leave_out_patient, f_n + '_fake_B.npy'))
            real_e_spec = []
            for j in range(real_e_temporal.shape[0]):
                e_spec = librosa.stft(real_e_temporal[j], n_fft=conf.eeg_n_fft, win_length=conf.eeg_win_len,
                                      hop_length=conf.eeg_hop)
                e_mag = np.log(np.abs(e_spec) + conf.epsilon)[: conf.w]
                e_angle = np.angle(e_spec)[: conf.w]
                e_IF = phase_operation.instantaneous_frequency(e_angle, time_axis=1)
                if is_IF:
                    e_spec = np.stack((e_mag, e_IF), axis=0)
                else:
                    e_spec = np.stack((e_mag, e_angle), axis=0)
                real_e_spec.append(e_spec)
            real_e_spec = normalizer.normalize(torch.from_numpy(np.asarray(real_e_spec)), 'eeg').numpy()[0]
            real_e_mag = real_e_spec[0]
            if is_IF:
                real_e_IF = real_e_spec[1]
            else:
                real_e_IF = phase_operation.instantaneous_frequency(real_e_spec[1], time_axis=1)
            if is_ae:
                fake_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_seeg.npy'))
            else:
                fake_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_B.npy'))
            real_s_spec = librosa.stft(real_s_temporal[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                                       hop_length=conf.seeg_hop)[: conf.w]
            e_IF = phase_operation.instantaneous_frequency(e_angle, time_axis=1)[: conf.w]
            if is_IF:
                real_s_spec = np.stack((np.log(np.abs(real_s_spec) + conf.epsilon),
                                        phase_operation.instantaneous_frequency(np.angle(real_s_spec), time_axis=1)), axis=0)
            else:
                real_s_spec = np.stack((np.log(np.abs(real_s_spec) + conf.epsilon), np.angle(real_s_spec)), axis=0)
            real_s_spec = normalizer.normalize(torch.from_numpy(np.asarray(real_s_spec[np.newaxis, :])), 'seeg').numpy()[0]
            real_s_mag = real_s_spec[0]
            if is_IF:
                real_s_IF = real_s_spec[1]
            else:
                real_s_IF = phase_operation.instantaneous_frequency(real_s_spec[1], time_axis=1)

            fake_s_temporal = IF_to_eeg(fake_s_spec, normalizer, iseeg=False, is_IF=True)
            fake_s_spec = librosa.stft(fake_s_temporal[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
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
        elif is_temporal:
            # real_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_real_seeg.npy'))
            # fake_s_spec = np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_seeg.npy'))
            # real_s_temporal = normalizer.denormalize_temporal(np.load(os.path.join(exper_dir, 'npys', f_n + '_real_seeg.npy')))
            real_e_spec = librosa.stft(real_e_temporal[0], n_fft=conf.eeg_n_fft, win_length=conf.eeg_win_len,
                                       hop_length=conf.eeg_hop)
            real_e_mag = np.log(np.abs(real_e_spec) + conf.epsilon)[: conf.w]
            real_e_angle = np.angle(real_e_spec)
            real_e_IF = phase_operation.instantaneous_frequency(real_e_angle, time_axis=1)[: conf.w]

            real_s_spec = librosa.stft(real_s_temporal[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                                       hop_length=conf.seeg_hop)[: conf.w]
            real_s_mag = np.log(np.abs(real_s_spec) + conf.epsilon)
            real_s_angle = np.angle(real_s_spec)
            real_s_IF = phase_operation.instantaneous_frequency(real_s_angle, time_axis=1)[: conf.w]

            fake_s_temporal = normalizer.denormalize_temporal(np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_seeg.npy')))[None, :]
            fake_s_spec = librosa.stft(fake_s_temporal[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                                       hop_length=conf.seeg_hop)
            fake_s_mag = np.log(np.abs(fake_s_spec) + conf.epsilon)[: conf.w]
            fake_s_angle = np.angle(fake_s_spec)
            fake_s_IF = phase_operation.instantaneous_frequency(fake_s_angle, time_axis=1)[: conf.w]
        if is_phase:
            plot_phase_test(np.load(os.path.join(exper_dir, 'npys', f_n + '_real_B.npy'))[0],
                            np.load(os.path.join(exper_dir, 'npys', f_n + '_fake_B.npy'))[0],
                            real_s_temporal, normalizer, conf,
                            temporal_save_path=os.path.join(temporal_save_dir, f_n + '_overlap' + '.png'),
                            mag_save_path=os.path.join(mag_save_dir, f_n + '.png'))
        else:
            draw_comparable_figure('Time(1/64 s)', 'Magnitude', conf.seeg_sf, s_start=0, s_end=None, ch_intv=0, show=False,
                                   save_path=os.path.join(temporal_save_dir, f_n + '.png'),
                                   Real_EEG=real_e_temporal, Real_SEEG=real_s_temporal, Fake_SEEG=fake_s_temporal)
            plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
                             vmin=-0.9, vmax=0.9, save_path=os.path.join(mag_save_dir,  f_n + ' Spectrogram clipped'),
                             RealEEG=real_e_mag, RealSEEG=real_s_mag, FakeSEEG=fake_s_mag)
            plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
                              vmin=None, vmax=None, save_path=os.path.join(mag_save_dir,  f_n + ' Spectrogram'),
                             RealEEG=real_e_mag, RealSEEG=real_s_mag, FakeSEEG=fake_s_mag)
            if not is_pghi:
                plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
                             vmin=None, vmax=None, save_path=os.path.join(mag_save_dir,  f_n + ' Phase clipped'),
                             RealEEG=real_e_IF, RealSEEG=real_s_IF, FakeSEEG=fake_s_IF)

        # plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
        #                  f_n + ' Spectrogram norm', vmin=-0.9, vmax=0.9, save_dir=mag_save_dir,
        #                  RealEEG=(real_e_mag - np.mean(real_e_mag)) / np.std(real_e_mag), RealSEEG=(real_s_mag - np.mean(real_s_mag))/np.std(real_s_mag), FakeSEEG=(fake_s_mag-np.mean(fake_s_mag))/np.std(fake_s_mag))


        # plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w), real_s_spec[0], title=f_n + ' Real SEEG',
        #                  save_dir=mag_save_dir, vmin=-.9, vmax=.9)
        # plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w), fake_s_spec[0], title=f_n + ' Fake SEEG',
        #                  save_dir=mag_save_dir, vmin=-.9, vmax=.9)
        # plot_spectrogram(np.arange(0, conf.w), np.arange(0, conf.seeg_ceiling_freq, conf.seeg_ceiling_freq / conf.w),
        #                  real_e_spec[0], title='avg Real EEG ' + str(n), save_dir=os.path.join(result_dir, 'mag'), vmin=-.9,
        #                  vmax=.9)
