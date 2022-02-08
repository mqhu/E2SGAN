#%%

from PIL import Image
from PGGAN import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import librosa.display

import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import h5py

import matplotlib.pyplot as plt
import spec_ops as spec_ops
import phase_operation as phase_op
import spectrograms_helper as spec_helper
from IPython.display import Audio
from normalizer import DataNormalizer

from tqdm import tqdm


g_net = Generator(256, 256, 2, is_tanh=True,channel_list=[256,256,256,256,256,128,64,32])


# g_checkpoint = torch.load('output4/Gnet_128x1024_step50.pth')
g_checkpoint = torch.load('LR2e-4_pitch_weight_10/Gnet_128x1024_step150.pth')

g_net.load_state_dict(g_checkpoint)
g_net.net_config = [6, 'stable', 1]
g_net.cuda()


fake_seed = torch.randn(1, 256, 1, 1).cuda()

ad = output_file(g_net,fake_seed,pitch=42)
Audio(ad,rate=16000)

#199
list_audio = []
for i in range(40,85):
    ad = output_file(g_net,fake_seed,pitch=i)
    ad = ad[:6000]
    list_audio.append(ad)
list_audio = np.hstack(list_audio)
Audio(list_audio,rate=16000)

#199
list_audio = []
for i in range(40,85):
    ad = output_file(g_net,fake_seed,pitch=i)
    ad = ad[:6000]
    list_audio.append(ad)
list_audio = np.hstack(list_audio)
Audio(list_audio,rate=16000)

#150
list_audio = []
for i in range(40,85):
    ad = output_file(g_net,fake_seed,pitch=i)
    ad = ad[:6000]
    list_audio.append(ad)
list_audio = np.hstack(list_audio)
Audio(list_audio,rate=16000)


def output_file(model,faked_seed, pitch):
    fake_pitch_label = torch.LongTensor(1, 1).random_() % 128
    pitch = [[pitch]]
    fake_pitch_label = torch.LongTensor(pitch)
    fake_one_hot_pitch_condition_vector = torch.zeros(1, 128).scatter_(1, fake_pitch_label, 1).unsqueeze(2).unsqueeze(3).cuda()
    fake_pitch_label = fake_pitch_label.cuda().squeeze()
    # generate random vector
#     fake_seed = torch.randn(1, 256, 1, 1).cuda()
    fake_seed_and_pitch_condition = torch.cat((fake_seed, fake_one_hot_pitch_condition_vector), dim=1)
    output = model(fake_seed_and_pitch_condition)
    output = output.squeeze()  # squeeze掉batch那一维

    spec = output[0].data.cpu().numpy().T
    IF = output[1].data.cpu().numpy().T
    spec, IF = denormalize(spec, IF, s_a=0.060437, s_b=0.034964, p_a=0.0034997, p_b=-0.010897)
    back_mag, back_IF = spec_helper.melspecgrams_to_specgrams(spec, IF)
    back_mag = np.vstack((back_mag,back_mag[1023]))
    back_IF = np.vstack((back_IF,back_IF[1023]))
    audio = mag_plus_phase(back_mag,back_IF)
    return audio


def denormalize(spec, IF, s_a, s_b, p_a, p_b):
    spec = (spec -s_b) / s_a
    IF = (IF-p_b) / p_a
    return spec, IF


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

def mag_plus_phase(mag, IF):

    mag =  np.exp(mag) - 1.0e-6 #log形式还原回去
    reconstruct_magnitude = np.abs(mag)

    # mag =  np.exp(mag) - 1e-6
    # reconstruct_magnitude = np.abs(mag)


    reconstruct_phase_angle = np.cumsum(IF * np.pi, axis=1) #按时间轴累加(因为IF是增量)
    stft = polar2rect(reconstruct_magnitude, reconstruct_phase_angle)
    inverse = librosa.istft(stft, hop_length = 512, win_length=2048, window = 'hann')

    return inverse
