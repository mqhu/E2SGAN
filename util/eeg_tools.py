import mne
import os
import numpy as np
from .numpy_tools import load_from_ndarray, save_as_ndarray, make_dataset, draw_comparable_figure
from scipy.stats import wasserstein_distance
import random
import shutil
import librosa
# import yasa
import matplotlib.pyplot as plt
#import pyedflib
import scipy
import scipy.io as io
#import matlab.engine
import ntpath
from gansynth import phase_operation


seeg_ch = ['EEG A1-Ref-0', 'EEG A2-Ref-0', 'POL A3', 'POL A4', 'POL A5', 'POL A6', 'POL A7', 'POL A8', 'POL A9',
           'POL A10', 'POL A11', 'POL A12', 'POL A13', 'POL A14', 'POL A15', 'POL A16', 'POL B1', 'POL B2', 'POL B3',
           'POL B4', 'POL B5', 'POL B6', 'POL B7', 'POL B8', 'POL B9', 'POL B10', 'POL B11', 'POL B12', 'POL B13',
           'POL B14', 'EEG C1-Ref', 'EEG C2-Ref', 'EEG C3-Ref-0', 'EEG C4-Ref-0', 'EEG C5-Ref', 'EEG C6-Ref', 'POL C7',
           'POL C8', 'POL C9', 'POL C10', 'POL C11', 'POL C12', 'POL C13', 'POL C14', 'POL D1', 'POL D2', 'POL D3',
           'POL D4', 'POL D5', 'POL D6', 'POL D7', 'POL D8', 'POL D9', 'POL D10', 'POL D11', 'POL D12', 'POL D13',
           'POL D14', 'POL D15', 'POL D16', 'POL E1', 'POL E2', 'POL E3', 'POL E4', 'POL E5', 'POL E6', 'POL E7',
           'POL E8', 'POL E9', 'POL E10', 'POL E11', 'POL E12', 'POL E13', 'POL E14', 'EEG F1-Ref', 'EEG F2-Ref',
           'EEG F3-Ref-0', 'EEG F4-Ref-0', 'EEG F5-Ref', 'EEG F6-Ref', 'EEG F7-Ref-0', 'EEG F8-Ref-0', 'EEG F9-Ref',
           'EEG F10-Ref', 'POL F11', 'POL F12', 'POL G1', 'POL G2', 'POL G3', 'POL G4', 'POL G5', 'POL G6', 'POL G7',
           'POL G8', 'POL G9', 'POL G10', 'POL G11', 'POL G12', 'POL H1', 'POL H2', 'POL H3', 'POL H4', 'POL H5',
           'POL H6', 'POL H7', 'POL H8', 'POL H9', 'POL H10', 'POL H11', 'POL H12', 'POL H13', 'POL H14', 'POL H15',
           'POL H16', 'POL K1', 'POL K2', 'POL K3', 'POL K4', 'POL K5', 'POL K6', 'POL K7', 'POL K8', 'POL K9',
           'POL K10', 'POL K11', 'POL K12', 'POL K13', 'POL K14', 'POL K15', 'POL K16']
eeg_ch = ['EEG Fp1-Ref', 'EEG F7-Ref-1', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG O1-Ref', 'EEG F3-Ref-1', 'EEG C3-Ref-1',
          'EEG P3-Ref', 'EEG FZ-Ref', 'EEG CZ-Ref', 'EEG PZ-Ref', 'EEG OZ-Ref', 'EEG Fp2-Ref', 'EEG F8-Ref-1',
          'EEG T4-Ref', 'EEG T6-Ref', 'EEG O2-Ref', 'EEG F4-Ref-1', 'EEG C4-Ref-1', 'EEG P4-Ref', 'EEG A1-Ref-1',
          'EEG A2-Ref-1']

exclusions = ['EEG OZ-Ref', 'EEG T3-Ref', 'EEG T6-Ref', 'EEG A1-Ref-1', 'EEG A2-Ref-1']

# tll pairs [('FZ', 'I8', array([69.34300274])), ('F4', 'A8', array([73.33140372])), ('PZ', 'E2', array([50.0013947]))]
# lk pairs [('Fp2', 'E10', array([20.69592009])), ('FZ', 'F1', array([24.33117395])), ('F4', 'F11', array([9.40280172])), ('F8', 'B14', array([14.80064537])), ('C4', 'D15', array([34.86986275])), ('T4', 'H14', array([19.30499903]))]
# zxl pairs [('CZ', 'D1', array([28.5934427])), ('PZ', 'C1', array([20.97461499])), ('O1', 'L1', array([46.05459558]))]
# yjh pairs [('F3', 'E10', array([44.71298704])), ('CZ', 'L3', array([69.89194194])), ('C3', 'G10', array([39.84213185]))]
# lmk pairs [('CZ', 'A14', array([9.39597045])), ('C3', 'B10', array([28.10216665])), ('T3', 'F3', array([38.61271169])), ('P4', 'H14', array([14.89213845]))]
# lxh pairs [('CZ', 'L17', array([20.55346079])), ('C3', 'H8', array([8.27186601])), ('T3', 'H3', array([56.06507585])), ('PZ', 'G10', array([21.1602206]))]
# wzw pairs [('PZ', 'G4', array([55.29771476])), ('O2', 'K2', array([41.76180341]))]

# eeg_sf = 64
# eeg_n_fft = 512
# eeg_win_len = 512
# eeg_hop = 8
# eeg_ceiling_freq = 28
# seeg_sf = 64
# seeg_n_fft = 512
# seeg_win_len = 512
# seeg_hop = 8
# seeg_ceiling_freq = 28
# epsilon = 1.0e-6

'''以上应该重构成一个class！！！'''
class Configuration:

    def __init__(self, transform='specgrams'):

        self.signal_len = 1024  # 1536
        self.eeg_sf = 64
        self.eeg_n_fft = 256  # 512
        self.eeg_win_len = 256
        self.eeg_hop = 8  # 12  # 8
        self.eeg_ceiling_freq = 32  # 28
        self.seeg_sf = 64
        self.seeg_n_fft = 256  # 512
        self.seeg_win_len = 256
        self.seeg_hop = 8  # 12  # 8
        self.seeg_ceiling_freq = 32  # 28
        self.epsilon = 1.0e-7
        self.eeg_pos = {'Fp1': (-1, 2), 'Fp2': (1, 2), 'F7': (-2, 1), 'F3': (-1, 1), 'FZ': (0, 1), 'F4': (1, 1),
                             'F8': (2, 1), 'T3': (-2, 0), 'C3': (-1, 0), 'CZ': (0, 0), 'C4': (1, 0), 'T4': (2, 0),
                             'T5': (-2, -1), 'P3': (-1, -1), 'PZ': (0, -1), 'P4': (1, -1), 'T6': (2, -1), 'O1': (-1, -2),
                             'O2': (1, -2)}
        self.seeg_mapping = {'lk': {'F1': 'FZ', 'E10': 'Fp2', 'F11': 'F4', 'B14': 'F8', 'D15': 'C4', 'H14': 'T4'},
                         'tll': {'B15': 'F7', 'K14': 'T3', 'M15': 'T4', 'E2': 'PZ'},
                         'zxl': {'G13': 'F8', 'D1': 'CZ', 'I10': 'C4', 'H14': 'T4', 'C1': 'PZ', 'F16': 'T6', 'L5': 'O2'},
                         'yjh': {'E15': 'F7', 'K14': 'F8', 'L9': 'C4', 'D15': 'T3', 'J14': 'T4'}}  # 建立与EEG坐标的映射
        self.h = 128  # 224
        self.w = 128  # 224
        if transform == 'specgrams':
            self.audio_length = (self.h - 1) * self.seeg_hop
        elif transform == 'pghi':
            self.audio_length = self.h * self.seeg_hop

conf = Configuration()

def filter_signal(raw_data, low, high, rm_pln=False):
    """

    :param raw_data: instance of mne.Raw
    :param low: float | None  the lower pass-band edge. If None the data are only low-passed.
    :param high: float | None  the upper pass-band edge. If None the data are only high-passed.
    :param rm_pln: bool  remove power line noise if True
    :return: instance of mne.Raw
    """
    if rm_pln:
        raw_data.notch_filter(np.arange(50., int(high / 50) * 50 + 1., 50), fir_design='firwin')
    raw_data.filter(low, high, fir_design='firwin')
    return raw_data


def down_sampling(raw_data, sfreq):
    """

    :param raw_data: instance of mne.Raw
    :param sfreq: New sample rate to use.
    :return: instance of mne.Raw
    """
    raw_data.resample(sfreq, npad="auto")
    return raw_data


def read_raw_signal(filename):

    if filename.endswith(".edf"):
        return mne.io.read_raw_edf(filename, preload=True)
    elif filename.endswith(".fif"):
        return mne.io.read_raw_fif(filename, preload=True)


def get_channels(raw_data):

    return raw_data.ch_names


def pick_data(raw_data, isA):

    chans = get_channels(raw_data)
    if isA:
        half = chans[:int(len(chans) / 2)]
    else:
        half = chans[int(len(chans) / 2)+1 :]
    picked_data = raw_data.pick_channels(half)
    print("Channels picked!")

    return picked_data[:, :][0]


def change_dir(path, isA, istrain=True):

    data_path = path
    if isA:
        data_path = os.path.join(data_path, 'A')
    else:
        data_path = os.path.join(data_path, 'B')
    if istrain:
        data_path = os.path.join(data_path, 'train')
    else:
        data_path = os.path.join(data_path, 'test')
    os.chdir(data_path)
    print("Current working directory is %s" % os.getcwd())


"""先原始数据滤波存为fif，然后再载入后存为npy格式，再以这种格式来crop，即直接对矩阵操作
记得处理好一个保存一个并删除占用的内存，免得内存泄漏"""


def get_next_number():
    numbers = sorted([int(os.path.splitext(os.path.basename(file))[0]) for file in make_dataset(os.getcwd())])
    if len(numbers) == 0:
        last_file_number = -1
    else:
        last_file_number = numbers[-1]

    return last_file_number + 1


def slice_data(raw_data, save_path, width, hop=None, start=0, end=None, start_number=None, prefix=''):
    """
    将脑电数组切片
    :param raw_data: mne.Raw，原始脑电
    :param save_path: 保存路径
    :param width: 划分片段长度
    :param hop: 片段间隔长度
    :param start: 划分起始点
    :param end: 划分结束点，若不提供则默认划分到终点
    :return: next: 下一个文件编号
    """

    origin_wd = os.getcwd()
    os.chdir(save_path)

    if end is None:
        end = raw_data.get_data().shape[1]

    if hop is None:
        hop = width

    if start_number is not None:
        next = start_number
    else:
        next = get_next_number()
    raw_data = raw_data.get_data()

    for i in range(start, end, hop):
        if i + width > end:
            break

        # assert not os.path.exists(str(next) + ".npy"), "File %s.npy already exists!" % str(next)
        segment = raw_data[:, i: i + width]
        save_as_ndarray(prefix + str(next), segment)
        if next % 1000 == 0:
            print("Slice %d done!" % next)
        next += 1

    print("Total pieces: %d" % next)
    os.chdir(origin_wd)

    return next


def slice_random_data(raw_data, save_path, random_starts, width, prefix=''):

    total = 0
    # end = raw_data.get_data().shape[1]
    raw_data = raw_data.get_data()

    for i, start in enumerate(random_starts):
        # if start + width > end:
        #     start = end - width
        segment = raw_data[:, start: start + width]
        save_as_ndarray(os.path.join(save_path, prefix + str(start)), segment)
        if total % 1000 == 0:
            print("Slice %d done!" % total)
        total += 1

    print("Total pieces: %d" % total)

    return total


def copy_random_files(n_random, seeg_src, seeg_dest, eeg_src, eeg_dest):
    dataset = make_dataset(seeg_src)
    rd = random.sample(range(0, len(dataset)), n_random)
    # t_no = rd[: int(n_random / 2)]
    # v_np = rd[int(n_random / 2):]
    for i, f in enumerate(dataset):
        if i in rd:
            f_n = os.path.basename(f)
            shutil.move(os.path.join(eeg_src, f_n), os.path.join(eeg_dest, f_n))
            shutil.move(os.path.join(seeg_src, f_n), os.path.join(seeg_dest, f_n))
        # if i in t_no:
        #     shutil.copy(os.path.join(src, "eeg", str(i)+'.npy'), os.path.join(dest, "B", "test"))
        #     shutil.copy(os.path.join(src, "seeg", str(i) + '.npy'), os.path.join(dest, "A", "test"))
        # elif i in v_np:
        #     shutil.copy(os.path.join(src, "eeg", str(i) + '.npy'), os.path.join(dest, "B", "val"))
        #     shutil.copy(os.path.join(src, "seeg", str(i) + '.npy'), os.path.join(dest, "A", "val"))
        # else:
        #     shutil.copy(os.path.join(src, "eeg", str(i) + '.npy'), os.path.join(dest, "B", "train"))
        #     shutil.copy(os.path.join(src, "seeg", str(i) + '.npy'), os.path.join(dest, "A", "train"))


def get_signals_by_variance(sig_dir, top_n, decending=True, style='average'):
    """

    :param sig_dir:
    :param top_n:
    :param decending:
    :param style:  option "average" or "single", select top variance according single or average channels
    :return: list of ndarray
    """
    file_paths = make_dataset(sig_dir)
    v = []
    signals = []
    for path in file_paths:
        signal = load_from_ndarray(path)
        signals.append(signal)
        if style == 'average':
            v.append(signal.var(axis=1).mean())
        elif style == 'single':
            v.append(signal.var(axis=1).max())

    sorted_signals = [pair[1] for pair in sorted(zip(v, signals), reverse=decending)]
    return sorted_signals[: top_n]


def get_time_freq_by_band(signal, low, high, iseeg, is_IF=False):
    '''
    filter signals
    :param signal: ndarray, shape (n_times)
    :param low: low frequency
    :param high: high frequency
    :return: filtered data
    '''
    # win_length = 512  win_length就默认等于n_fft吧
    if iseeg:
        sf = conf.eeg_sf
        n_fft = conf.eeg_n_fft
        hop = conf.eeg_hop
    else:
        sf = conf.seeg_sf
        n_fft = conf.seeg_n_fft
        hop = conf.seeg_hop

    #transferred = signal.astype(np.float64)
    if high < conf.seeg_sf / 2:
        filtered = mne.filter.filter_data(signal, sf, low, high)  # 时域信号
    else:
        filtered = signal
    spec = librosa.stft(filtered, n_fft=n_fft, hop_length=hop)[int(low * n_fft / sf): int(high * n_fft / sf)]  # 频域只取关心波段的值
    magnitude = np.abs(spec)  # + conf.epsilon
    phase = np.angle(spec)
    # if is_IF:
    IF = phase_operation.instantaneous_frequency(phase, time_axis=1)
    # phase = np.where(abs(phase) < 3.14, 0, np.sign(phase) * 1) #将派变为1，负派变为-1

    return filtered, magnitude, phase, IF


def time_frequency_transform(raw_data, freqs, sf, output):
    """

    :param raw_data: array
    :param freqs: array_like of float, shape (n_freqs,)  list of output frequencies
    :param output: str in [‘complex’, ‘phase’, ‘power’, 'avg_power', 'avg_power_itc' ]
    :return: Time frequency transform of epoch_data.  If output is in  [‘complex’, ‘phase’, ‘power’], \
    then shape of out is (n_epochs, n_chans, n_freqs, n_times), else it is (n_chans, n_freqs, n_times).
    """

    if 'avg' not in output:
        raw_data = raw_data[np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(raw_data, sfreq=sf,
                                                freqs=freqs, n_cycles=freqs / 2.,
                                                output=output)
    return power


def analyze_power_each_freq(power, freqs, method):
    n_chan = len(power[0][0])
    n_freq = len(freqs)
    result = [0 for _ in range(n_freq)]
    for chan in power[0]:
        for i in range(n_freq):
            if method == 'mean':
                result[i] += chan[i].mean()
            elif method == 'max':
                result[i] = max(result[i], chan[i].max())
    if method =='mean':
        result = [1.0 * i / n_chan for i in result]

    return result


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form.
        复数极坐标形式变成直角坐标系"""
    #     mag = np.complex(mag)
    temp_mag = np.zeros(mag.shape,dtype=np.complex_)
    temp_phase = np.zeros(mag.shape,dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
    #             print(mag[i,j])
            temp_mag[i,j] = np.complex(mag[i,j])
    #             print(temp_mag[i,j])

    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i,j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))
    #             print(temp_mag[i,j])

    #     phase = np.complex(np.cos(phase_angle), np.sin(phase_angle))

    return temp_mag * temp_phase


def mag_plus_phase(mag, IF, iseeg, is_IF=False):
    '''记得和base_dataset里的__to_mag_and_IF统一！！！'''

    hop = conf.eeg_hop if iseeg else conf.seeg_hop
    n_fft = conf.eeg_n_fft if iseeg else conf.seeg_n_fft

    h, w = mag.shape
    # mag = np.exp(mag)
    mag = np.exp(mag) - conf.epsilon #log形式还原回去

    if h < n_fft // 2 + 1:
        mag = np.vstack((mag, np.zeros((n_fft // 2 - h + 1, w))))
        IF = np.vstack((IF, np.zeros((n_fft // 2 - h + 1, w))))
    reconstruct_magnitude = np.abs(mag)

    # mag =  np.exp(mag) - 1e-6
    # reconstruct_magnitude = np.abs(mag)

    # 若使用IF特征则需要下面这行，否则默认IF参数就是普通的相位值
    if is_IF:
        reconstruct_phase_angle = np.cumsum(IF * np.pi, axis=1) #按时间轴累加(因为IF是增量)
    else:
        reconstruct_phase_angle = IF
    stft = polar2rect(reconstruct_magnitude, reconstruct_phase_angle)
    inverse = librosa.istft(stft, hop_length=hop, window='hann')

    return inverse


def IF_to_eeg(IF_output, normalizer, iseeg=True, is_IF=False):
    '''
    把IF结果或者sftf结果变回原来的样子，可以是eeg也可以是seeg
    :param IF_output: 模型的输出
    :param normalizer: 归一化器
    :return: 还原的结果
    '''
    # for i in range(len(IF_output[0])):
    #     magnitude = np.vstack((IF_output[0][i], IF_output[0][i][-1]))
    #     IF = np.vstack((IF_output[1][i], IF_output[1][i][-1]))
    #     magnitude, IF = normalizer.denormalize(magnitude, IF)
    #     out = mag_plus_phase(magnitude, IF)
    #     fake.append(out)
    n_fft = conf.eeg_n_fft if iseeg else conf.seeg_n_fft
    h, w = IF_output.shape[-2:]
    if iseeg == False:
        IF_output = IF_output[np.newaxis, :]
    out = []
    for i in range(IF_output.shape[0]):
        if h < n_fft // 2:
            magnitude, IF = normalizer.denormalize(IF_output[i][0], IF_output[i][1], iseeg)
            # magnitude = np.vstack((magnitude, np.zeros((n_fft // 2 - h + 1, w))))
            # IF = np.vstack((IF, np.zeros((n_fft // 2 - h + 1, w))))
            # print(magnitude.shape)
        else:
            magnitude = np.vstack((IF_output[i][0], IF_output[i][0][-1]))
            IF = np.vstack((IF_output[i][1], IF_output[i][1][-1]))  # 不使用IF特征的话这里就是普通的相位
            magnitude, IF = normalizer.denormalize(magnitude, IF, iseeg)
        out.append(mag_plus_phase(magnitude, IF, iseeg, is_IF))

    recovered = np.asarray(out)

    return recovered

'''
def calculate_spindle_precision(real_path, fake_path, normalizer):
    dataset = make_dataset(real_path)
    f_names = []
    sp_real = 0
    sp_fake = 0
    n_chan = 1
    sp_list = []
    for f_name in dataset:
        f_names.append(os.path.basename(f_name).split('.')[0])
    sf = 128.
    for f_n in f_names:
        for i in range(n_chan):
            real_B = np.load(real_path + f_n + '.npy')
            fake_B = np.load(fake_path + f_n + '_fake_B.npy')
            fake_B = IF_to_eeg(fake_B, normalizer)[None, :]
            spp1 = yasa.spindles_detect(real_B[i] * 1e6, sf)
            spp2 = yasa.spindles_detect(fake_B[i] * 1e6, sf)
            if spp1 is not None:
                sp_real += 1
                if spp2 is not None:
                    sp_fake += 1
                    sp_list.append({'f_n': f_n, 'chan_n': i})
    if sp_real == 0:
        score = 0
    else:
        score = sp_fake / sp_real
    print('sp_real：' + str(sp_real))
    print('sp_fake：' + str(sp_fake))
    print('precision：' + str(score))

    return score'''


'''
def calculate_spindle(real_path, fake_path, ref_dir, normalizer):  # 采用白质内SEEG电极作为reference
    dataset = make_dataset(real_path)
    found1 = 0
    matched1 = 0
    found2 = 0
    matched2 = 0
    sf = 128.

    for f_name in dataset:
        f_n = os.path.basename(f_name).split('.')[0]
        real_B = np.load(os.path.join(real_path, f_n + '.npy'))[0]  # 转化成1维数据yasa使用
        fake_B = np.load(os.path.join(fake_path, f_n + '_fake_B.npy'))
        fake_B = IF_to_eeg(fake_B, normalizer)
        ref = np.load(os.path.join(ref_dir, f_n + '.npy'))[0]
        #ref2 = np.load(os.path.join(ref2_dir, f_n + '.npy'))[0]
        corrected_real_B = real_B - ref
        corrected_fake_B = fake_B - ref
        spp1 = yasa.spindles_detect(corrected_real_B * 1e6, sf)
        spp2 = yasa.spindles_detect(corrected_fake_B * 1e6, sf)
        if spp1 is not None:
            found1 += 1
            if spp2 is not None:
                matched1 += 1
        if spp2 is not None:
            found2 += 1
            if spp1 is not None:
                matched2 += 1
    if found1 == 0:
        recall = 0
    else:
        recall = matched1 / found1
    if found2 == 0:
        precision = 0
    else:
        precision = matched2 / found2
    print("precision=%d/%d=%f " % (matched2, found2, precision))
    print("recall=%d/%d=%f " % (matched1, found1, recall))

    return precision, recall
'''

'''
def modify_a7Init(eng, attr, value):
    eng.eval('DEF_a7.' + attr + '=\'' + value + '\'', nargout=0)
    # eng.eval('DEF_a7.inputPath=\'' + inputPath + '\'', nargout=0)
    # eng.eval('DEF_a7.EEGvector=\'' + EEGvector + '\'', nargout=0)
    # eng.eval('DEF_a7.sleepStaging=\'' + sleepStaging + '\'', nargout=0)
    # eng.eval('DEF_a7.artifactVector=\'' + artifactVector + '\'', nargout=0)
    # eng.eval('DEF_a7.outputGrandParentDir=\'' + outputGrandParentDir + '\'', nargout=0)


def npy_to_mat(npy_name, label, mat_dir):

    data = np.load(npy_name)
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    io.savemat(os.path.join(mat_dir, os.path.basename(npy_name).split('.')[0]) + '.mat', {label: data})


def calculate_a7_spindles(real_inputPath, fake_inputPath, real_outputGrandParentDir, fake_outputGrandParentDir):
    dataset = make_dataset(real_inputPath)
    found1 = 0
    matched1 = 0
    found2 = 0
    matched2 = 0

    eng = matlab.engine.start_matlab()
    eng.initA7_DEF(nargout=0)

    for f_name in dataset:
        f_n = os.path.basename(f_name).split('.')[0]
        modify_a7Init(eng, 'inputPath', real_inputPath)
        modify_a7Init(eng, 'EEGvector', f_n + '.mat')
        modify_a7Init(eng, 'outputGrandParentDir', real_outputGrandParentDir)
        eng.a7MainScript(nargout=0)

        # 看看读取spindle结果

        modify_a7Init(eng, 'inputPath', fake_inputPath)
        modify_a7Init(eng, 'EEGvector', f_n + 'fake_B.mat')
        modify_a7Init(eng, 'outputGrandParentDir', fake_outputGrandParentDir)
        eng.a7MainScript(nargout=0)

        with open("/home/cbd109/Users/Data/hmq/GANDatasets/testmatlab/output/20200825_152014_EventDetection.txt",
                  'r') as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                print(line.split('\t'))'''


def save_comparison_plots(real_dir, fake_dir, save_dir, normalizer):

    dataset = make_dataset(real_dir)
    for f_name in dataset:
        f_n = os.path.basename(f_name).split('.')[0]
        real_B = np.load(os.path.join(real_dir, f_n + '.npy'))
        fake_B = np.load(os.path.join(fake_dir, f_n + '_fake_B.npy'))
        fake_B = IF_to_eeg(fake_B, normalizer)

        draw_comparable_figure(Real=real_B, Fake=fake_B[None, :], ch_intv=0, show=False)
        plt.savefig(os.path.join(save_dir, f_n))
        plt.close()


def save_origin_mat(visuals, image_dir, image_path, normlizer):
    """
    这个是用来把测试的数据保存为mat格式给a7用的，在运行test.py时用
    :param visuals:
    :param image_dir:
    :param image_path:
    :param save_size:
    :return:
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for label, im_data in visuals.items():
        if 'fake' in label:
            mat_name = '%s_%s' % (name, label) # im_data是tensor
            mat_data = im_data[0]
            mat_data = mat_data.cpu().numpy()
            mat_data = IF_to_eeg(mat_data, normlizer)
            mat_data = np.expand_dims(mat_data, axis=0)
            save_path = os.path.join(image_dir, mat_name)
            io.savemat(save_path, {'EEGvector': mat_data})


# def save_raw_as_edf(raw, fout_name):  # 把raw数据存为edf格式
#     '''
#     :param raw:  mne.io.raw格式数据
#     :param fout_name:   输出的文件名
#     :return:
#     '''
#     NChannels = raw.info['nchan']
#     channels_info = list()
#     for i in range(NChannels):
#         '''默认参数来自edfwriter.py'''
#         ch_dict = dict()
#         ch_dict['label'] = raw.info['chs'][i]['ch_name']
#         ch_dict['dimension'] = 'mV'
#         ch_dict['sample_rate'] = raw.info['sfreq']
#         ch_dict['physical_max'] = 1.0
#         ch_dict['physical_min'] = -1.0
#         ch_dict['digital_max'] = 32767
#         ch_dict['digital_min'] = -32767
#         ch_dict['transducer'] = 'trans1'
#         ch_dict['prefilter'] = "pre1"
#         channels_info.append(ch_dict)
#
#     fileOut = os.path.join('.', fout_name + '.edf')
#     fout = pyedflib.EdfWriter(fileOut, NChannels, file_type=pyedflib.FILETYPE_EDFPLUS)
#     data_list, _ = raw[:, :]
#     #print(data_list)
#     fout.setSignalHeaders(channels_info)
#     fout.writeSamples(data_list)
#     fout.close()
    #print("Done!")
    #del fout
    #del data_list

def pghi_invert(output, normalizer, postprocessor, iseeg=True):

    output = output.squeeze()

    magnitude = normalizer.denormalize(output, is_eeg=iseeg)
    recovered = postprocessor(np.vstack((magnitude, magnitude[-1])))[: conf.signal_len]

    return recovered[np.newaxis, :]
