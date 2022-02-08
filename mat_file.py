import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from gansynth.normalizer import DataNormalizer
from util.distance_metrics import get_neighbors_distance, EUD_interpolate, calculate_distance, Hellinger_Distance, Bhattacharyya_Distance
from util.numpy_tools import make_dataset
import csv
import mne
from util.eeg_tools import Configuration

# seeg_ch = ['EEG A1-Ref-0', 'EEG A2-Ref-0', 'POL A3', 'POL A4', 'POL A5', 'POL A6', 'POL A7', 'POL A8', 'POL A9',
#            'POL A10', 'POL A11', 'POL A12', 'POL A13', 'POL A14', 'POL A15', 'POL A16', 'POL B1', 'POL B2',
#            'POL B3',
#            'POL B4', 'POL B5', 'POL B6', 'POL B7', 'POL B8', 'POL B9', 'POL B10', 'POL B11', 'POL B12', 'POL B13',
#            'POL B14', 'EEG C1-Ref', 'EEG C2-Ref', 'EEG C3-Ref-0', 'EEG C4-Ref-0', 'EEG C5-Ref', 'EEG C6-Ref',
#            'POL C7',
#            'POL C8', 'POL C9', 'POL C10', 'POL C11', 'POL C12', 'POL C13', 'POL C14', 'POL D1', 'POL D2', 'POL D3',
#            'POL D4', 'POL D5', 'POL D6', 'POL D7', 'POL D8', 'POL D9', 'POL D10', 'POL D11', 'POL D12', 'POL D13',
#            'POL D14', 'POL D15', 'POL D16', 'POL E1', 'POL E2', 'POL E3', 'POL E4', 'POL E5', 'POL E6', 'POL E7',
#            'POL E8', 'POL E9', 'POL E10', 'POL E11', 'POL E12', 'POL E13', 'POL E14', 'EEG F1-Ref', 'EEG F2-Ref',
#            'EEG F3-Ref-0', 'EEG F4-Ref-0', 'EEG F5-Ref', 'EEG F6-Ref', 'EEG F7-Ref-0', 'EEG F8-Ref-0', 'EEG F9-Ref',
#            'EEG F10-Ref', 'POL F11', 'POL F12', 'POL G1', 'POL G2', 'POL G3', 'POL G4', 'POL G5', 'POL G6',
#            'POL G7',
#            'POL G8', 'POL G9', 'POL G10', 'POL G11', 'POL G12', 'POL H1', 'POL H2', 'POL H3', 'POL H4', 'POL H5',
#            'POL H6', 'POL H7', 'POL H8', 'POL H9', 'POL H10', 'POL H11', 'POL H12', 'POL H13', 'POL H14', 'POL H15',
#            'POL H16', 'POL K1', 'POL K2', 'POL K3', 'POL K4', 'POL K5', 'POL K6', 'POL K7', 'POL K8', 'POL K9',
#            'POL K10', 'POL K11', 'POL K12', 'POL K13', 'POL K14', 'POL K15', 'POL K16']
# eeg_ch = ['EEG Fp1-Ref', 'EEG F7-Ref-1', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG O1-Ref', 'EEG F3-Ref-1', 'EEG C3-Ref-1',
#           'EEG P3-Ref', 'EEG FZ-Ref', 'EEG CZ-Ref', 'EEG PZ-Ref', 'EEG OZ-Ref', 'EEG Fp2-Ref', 'EEG F8-Ref-1',
#           'EEG T4-Ref', 'EEG T6-Ref', 'EEG O2-Ref', 'EEG F4-Ref-1', 'EEG C4-Ref-1', 'EEG P4-Ref']

eeg_coor = {'Fp1': (-21.5, 70.2, -0.1), "Fp2": (28.4, 69.1, -0.4),
             "Fz": (0.6, 40.9, 53.9), "F3": (-35.5, 49.4, 32.4),
             "F4": (40.2, 47.6, 32.1), "F7": (-54.8, 33.9, -3.5),
             "F8": (56.6, 30.8, -4.1), "Cz": (0.8, -14.7, 73.9),
             "C3": (-52.2, -16.4, 57.8), "C4": (54.1, -18.0, 57.5),
             "T3": (-70.2, -21.3, -10.7), "T4": (71.9, -25.2, -8.2),
             "Pz": (0.2, -62.1, 64.5), "P3": (-39.5, -76.3, 47.4),
             "P4": (36.8, -74.9, 49.2), "T5": (-61.5, -65.3, 1.1),
             "T6": (59.3, -67.6, 3.8), "O1": (-26.8, -100.2, 12.8),
             "O2": (24.1, -100.5, 14.1)}


def retrieve_chs_from_mat(filepath, stripped_seeg_path):

    pos_info = sio.loadmat(filepath)
    elec_pos = list()
    patient_name = os.path.basename(stripped_seeg_path).split('_')[0]
    if patient_name.upper() == 'LK':  #LK在.mat文件中第一个cell的名字不一样
        cell = 'elec_Info_Final_wm'
    else:
        cell = 'elec_Info_Final'
    pos_info = pos_info[cell]
    name_ls = pos_info['name'][0][0][0]
    norm_pos_ls = pos_info['norm_pos'][0][0][0]
    wm_label_ls = pos_info['ana_label_name'][0][0][0]
    stripped_seeg = np.load(stripped_seeg_path)
    size = name_ls.size
    for i in range(size):  #name为字符串,pos为ndarray格式
        name = name_ls[i][0]
        dc = {}
        tp = ""
        for ch in stripped_seeg:
            if name == ch:
                dc = {'name': ch, 'pos': norm_pos_ls[i][0]}
                tp = wm_label_ls[i][0].lower()
                break
        if 'ctx' in tp or 'cortex' in tp:
            c_or_w = 'ctx'
        elif 'wm' in tp or 'whitematter' in tp or 'white-matter' in tp:
            c_or_w = 'wm'
        else:
            c_or_w = 'unknown'
        dc['type'] = c_or_w
        elec_pos.append(dc)
    print("elec_pos length =", len(elec_pos))
    return elec_pos


def retrieve_chs_from_csv(filepath):

    with open(filepath, 'r') as f:
        reader = list(csv.reader(f))
    reader = list(zip(*reader))

    elec_pos = list()
    name_ls = reader[0]
    wm_label_ls = reader[1]
    norm_pos_ls = reader[2]
    size = len(name_ls)
    for i in range(size):  #name为字符串,pos为ndarray格式
        name = name_ls[i]
        pos = set(norm_pos_ls[i].split(' '))
        pos.remove('')
        pos = np.asarray([float(p) for p in pos])
        dc = {'name': name, 'pos': pos}
        tp = wm_label_ls[i].lower()
        if 'ctx' in tp or 'cortex' in tp:
            c_or_w = 'ctx'
        elif 'wm' in tp or 'whitematter' in tp or 'white-matter' in tp:
            c_or_w = 'wm'
        else:
            c_or_w = 'unknown'
        dc['type'] = c_or_w
        elec_pos.append(dc)
    print("elec_pos length =", len(elec_pos))
    return elec_pos


def get_nearest_pair():

    patient = 'wzw'
    info_dir = "/home/hmq/Infos/position_info/"
    eeg_pos = np.load(os.path.join(info_dir, 'eeg_pos.npy'), allow_pickle=True).item()
    seeg_info = np.load(os.path.join(info_dir, patient + '_seeg_pos.npy'), allow_pickle=True)
    eeg_chs = np.load(os.path.join("/home/hmq/Infos/ch_names/", patient + "_stripped_eeg.npy"), allow_pickle=True)
    # F1 = None
    # for item in seeg_info:
    #     if item['name'] == 'EEG F1-Ref':
    #         F1 = item['pos']
    # print(F1)  # [ 4.04258916 35.95939224 48.34240927]
    # seeg_info = np.load("/home/cbd109/Users/hmq/LK_info/LK_name_pos_wc.npy", allow_pickle=True).tolist()
    # eeg_seeg = {}
    # seeg_dist = {}
    pair_dist = []
    pairs = []
    # seeg_info = np.load("/home/hmq/Infos/position_info/LK_seeg_pos.npy", allow_pickle=True)
    # for d in seeg_info:
    #     seeg_dist[d['name']] = d['pos']
    # names, dist = get_neighbors_distance(eeg_pos, [4.04258916, 35.95939224, 48.34240927], 5)
    # print(names)
    # print(dist)

    '''eeg_pos = {}
    for k, v in eeg_coor.items():
        for ch in eeg_ch:
            index = ch.lower().find(k.lower())
            if index >= 0:
                eeg_pos[ch] = v'''
    filtered_eeg_pos = {}
    for k, v in eeg_pos.items():
        if k.upper() in eeg_chs:
            filtered_eeg_pos[k] = v
    print(filtered_eeg_pos)
    for k, v in filtered_eeg_pos.items():
        #thd = 30.
        e_pos = np.asarray(v)
        min_pos = None
        min = float('inf')
        idx = ''
        for s_ch in seeg_info:
            if s_ch['type'] != 'ctx':
                continue
            s_pos = s_ch['pos']
            #print(e_pos)
            #print(s_pos)
            dist = np.linalg.norm(e_pos - s_pos)
            if dist < min:
                min = dist
                idx = s_ch['name']
                min_pos = s_pos
        if idx == '':
            continue
        else:
            pairs.append((k, idx, min_pos))
            pair_dist.append(min)
    print(pairs)
    print(pair_dist)
    print(len(pairs))
    final_pair = []
    for (e_ch, s_ch, s_pos) in pairs:
        e_name_ls, dist = get_neighbors_distance(filtered_eeg_pos, s_pos, 1)
        if e_name_ls[0].lower() == e_ch.lower():
            final_pair.append((e_ch, s_ch, dist))
    print(final_pair)
    print(len(final_pair))


def sort_eeg_seeg_pair_by_distance():
    '''
    根据两种相似度来排序每个EEG电极最相似的SEEG电极
    :return:
    '''
    conf = Configuration()
    patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    eeg_pos = np.load("/home/hmq/Infos/position_info/eeg_pos.npy", allow_pickle=True).item()
    raw_dir = "/home/hmq/Signal/preprocessed/"
    pos_dir = "/home/hmq/Infos/position_info/"
    ch_dir = "/home/hmq/Infos/ch_names/"

    for p in patients[2:3]:
        seeg_info = np.load(os.path.join(pos_dir, p + '_seeg_pos.npy'), allow_pickle=True)
        raw_eeg = mne.io.read_raw_fif(os.path.join(raw_dir, p + '_eeg_raw.fif'))
        raw_seeg = mne.io.read_raw_fif(os.path.join(raw_dir, p + '_seeg_raw.fif'))
        s_psd_dict = {}
        e_psd_dict = {}
        e_s_h_dist = {}
        e_s_b_dist = {}

        for s_ch in seeg_info:
            if s_ch['type'] == 'ctx':
                s_ch_name = s_ch['name']
                seeg = raw_seeg[[s_ch_name], :][0][0]
                s_psd_dict[s_ch_name] = mne.time_frequency.psd_array_welch(seeg, conf.seeg_sf, fmin=0, fmax=28, n_fft=conf.seeg_n_fft)[0]

        for e_ch in raw_eeg.ch_names:
            eeg = raw_eeg[[e_ch], :][0][0]
            e_psd_dict[e_ch] = mne.time_frequency.psd_array_welch(eeg, conf.eeg_sf, fmin=0, fmax=28, n_fft=conf.eeg_n_fft)[0]

        for e_ch, e_psd in e_psd_dict.items():
            e_s_h_dist[e_ch] = {}
            e_s_b_dist[e_ch] = {}
            for s_ch, s_psd in s_psd_dict.items():
                h_d = Hellinger_Distance(s_psd, e_psd)
                b_d = Bhattacharyya_Distance(s_psd, e_psd)
                e_s_h_dist[e_ch][s_ch] = h_d
                e_s_b_dist[e_ch][s_ch] = b_d

        for k, v in e_s_h_dist.items():
            e_s_h_dist[k] = sorted(v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        for k, v in e_s_b_dist.items():
            e_s_b_dist[k] = sorted(v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        print('Patient {}: '.format(p))
        print(e_s_h_dist)
        print(e_s_b_dist)
        np.save("/home/hmq/Infos/dist/" + p + "_hd.npy", e_s_h_dist)
        np.save("/home/hmq/Infos/dist/" + p + "_bd.npy", e_s_b_dist)


def rank_by_distance():
    '''
    根据两种相似度的排序结果，给出平均后的rank
    :return:
    '''
    conf = Configuration()
    patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    similarity_dir = "/home/hmq/Infos/similarity/"

    for p in patients:
        rank = {}
        hd = np.load(os.path.join(similarity_dir, p + '_hd.npy'), allow_pickle=True).item()
        bd = np.load(os.path.join(similarity_dir, p + '_bd.npy'), allow_pickle=True).item()

        for e_ch in hd.keys():
            rank[e_ch] = {}
            l1 = [pair[0] for pair in hd[e_ch]]
            l2 = [pair[0] for pair in bd[e_ch]]
            for i, s_ch in enumerate(l1):
                rank1 = i + 1
                rank2 = l2.index(s_ch) + 1
                rank[e_ch][s_ch] = (rank1 + rank2) // 2

        for k, v in rank.items():
            rank[k] = sorted(v.items(), key=lambda kv: (kv[1], kv[0]))
            rank[k] = list(map(list, rank[k]))
            cnt = 1
            for i in range(len(rank[k]) - 1):  # 按照递增顺序整理一下rank
                pre = rank[k][i][1]
                rank[k][i][1] = cnt
                if pre != rank[k][i+1][1]:
                    cnt += 1

            rank[k][-1][1] = cnt

        print('Patient {}: '.format(p))
        print(rank)
        np.save("/home/hmq/Infos/similarity/" + p + "_final_rank.npy", rank)


# def Kendall_corr_statistics():
#     patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
#     similarity_dir = "/home/hmq/Infos/similarity/"
#
#     for p in patients:
#         h_d = np.load(os.path.join(similarity_dir, p + '_hd.npy'), allow_pickle=True).item()
#         b_d = np.load(os.path.join(similarity_dir, p + '_bd.npy'), allow_pickle=True).item()
#
#         for e_ch in h_d.keys():



if __name__ == '__main__':

    conf = Configuration()
    patients = ['lk', 'zxl', 'yjh', 'tll', 'lmk', 'lxh', 'wzw']
    eeg_pos = np.load("/home/hmq/Infos/position_info/eeg_pos.npy", allow_pickle=True).item()
    raw_dir = "/home/hmq/Signal/preprocessed/"
    pos_dir = "/home/hmq/Infos/position_info/"
    ch_dir = "/home/hmq/Infos/ch_names/"
    print(eeg_pos)

    for p in patients:
        seeg_info = np.load(os.path.join(pos_dir, p + '_seeg_pos.npy'), allow_pickle=True)
        eeg_chs = np.load(os.path.join(ch_dir, p + '_stripped_eeg.npy'), allow_pickle=True)
        s_pos_dict = {}
        e_s_dist = {}

        for s_ch in seeg_info:
            if s_ch['type'] == 'ctx':
                s_ch_name = s_ch['name']
                s_pos_dict[s_ch_name] = s_ch['pos']

        for e_ch in eeg_chs:
            if e_ch in eeg_pos.keys():
                e_name_ls, dist = get_neighbors_distance(s_pos_dict, eeg_pos[e_ch])
                e_s_dist[e_ch] = sorted(zip(e_name_ls, dist), key=lambda kv: (kv[1], kv[0]))

        print('Patient {}: '.format(p))
        print(e_s_dist)
        np.save("/home/hmq/Infos/similarity/" + p + "_pos_dist.npy", e_s_dist)
        for k, v in e_s_dist.items():
            e_s_dist[k] = list(map(list, v))
            for i in range(len(v)):
                e_s_dist[k][i][1] = i + 1
        print(e_s_dist)
        np.save("/home/hmq/Infos/similarity/" + p + "_pos_rank.npy", e_s_dist)
