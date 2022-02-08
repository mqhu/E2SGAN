from util.eeg_tools import *
from util.numpy_tools import *
import numpy as np


if __name__ == '__main__':

    conf = Configuration()
    excluded_eeg = []
    ordered_eeg = []
    clean_ordered_eeg = []
    patient = 'LK'
    order_save_path = "/home/hmq/Infos/ordered_eeg_ch/"

    for eeg in eeg_ch:
        if eeg not in exclusions:
            excluded_eeg.append(eeg)
    for clean_eeg in conf.eeg_pos.keys():
        for eeg in excluded_eeg:
            _eeg = eeg.split(' ')[1].split('-')[0]
            if _eeg.lower() == clean_eeg.lower():
                ordered_eeg.append(eeg)
                clean_ordered_eeg.append(clean_eeg)
                break
    # eeg = ['EEG FZ-Ref']
    # seeg = ['EEG F1-Ref']
    # interpolation = ['EEG FZ-Ref', 'EEG F4-Ref-1', 'EEG F3-Ref-1', 'EEG CZ-Ref', 'EEG Fp2-Ref']  # [ 8.19437855 41.31195912 44.70376615 56.83378265 63.77621662]
    # nearest_seeg = ['EEG F2-Ref', 'EEG F3-Ref-0', 'EEG F4-Ref-0', 'EEG F5-Ref', 'EEG A2-Ref-0', 'EEG A1-Ref-0',
    #                 'POL A3', 'EEG F6-Ref', 'POL A4', 'POL A5']
    itv = [(3600, 4320), (10800, 11520)]
    sf = 64
    # eeg_pos = np.load("/home/hmq/Infos/position_info/eeg_pos.npy", allow_pickle=True).item()
    # picked_pos = {}
    # picked_pos['EEG F1-Ref'] = [4.04258916, 35.95939224, 48.34240927]
    # for k, v in eeg_pos.items():
    #     for ch in interpolation:
    #         if k.lower() in ch.lower():
    #             picked_pos[ch] = list(v)
    raw = read_raw_signal("/home/hmq/Signals/preprocessed_LK/LK_Sleep_Aug_4th_2am_seeg_raw.fif")  # Sleep_Aug_4th_2am
    raw.pick_channels(['EEG F1-Ref'])
    # raw.reorder_channels(interpolation)
    # raw.resample(sf, npad="auto")
    # raw.pick_channels(ordered_eeg)
    # raw.reorder_channels(ordered_eeg)
    # raw.filter(1., None, fir_design='firwin')
    print(ordered_eeg)
    print(raw.info.ch_names)
    prefix = 'F1_'

    # raw1 = raw.copy().crop(tmin=0, tmax=3600)
    # next = slice_data(raw1, '/home/hmq/Datasets/LK_224x224_e2s_28Hz_middle_attn/A/train/', 1784, hop=1784 // 4, start_number=0, prefix=prefix)
    # raw1 = raw.copy().crop(tmin=3600, tmax=4320)
    # next = slice_data(raw1, '/home/hmq/Datasets/LK_224x224_e2s_28Hz_middle_attn/A/test/', 1784, hop=1784 // 4, start_number=next, prefix=prefix)
    # raw1 = raw.copy().crop(tmin=4320, tmax=10800)
    # next = slice_data(raw1, '/home/hmq/Datasets/LK_224x224_e2s_28Hz_middle_attn/A/train/', 1784, hop=1784 // 4,
    #                   start_number=next, prefix=prefix)
    # raw1 = raw.copy().crop(tmin=10800, tmax=11520)
    # next = slice_data(raw1, '/home/hmq/Datasets/LK_224x224_e2s_28Hz_middle_attn/A/test/', 1784, hop=1784 // 4,
    #                   start_number=next, prefix=prefix)
    # raw1 = raw.copy().crop(tmin=11520, tmax=None)
    # slice_data(raw1, '/home/hmq/Datasets/LK_224x224_e2s_28Hz_middle_attn/A/train/', 1784, hop=1784 // 4, start_number=next, prefix=prefix)
