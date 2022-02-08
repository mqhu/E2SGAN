import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from gansynth.pytorch_nsynth_lib.nsynth import NSynth
from IPython.display import Audio

import librosa
import librosa.display
from gansynth import phase_operation
from tqdm import tqdm
import h5py


import gansynth.spec_ops as spec_ops
import gansynth.phase_operation as phase_op
import gansynth.spectrograms_helper as spec_helper


train_data = h5py.File('../data/Nsynth_melspec_IF_pitch.hdf5', 'w')


# audio samples are loaded as an int16 numpy array
# rescale intensity range as float [-1, 1]
toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
# use instrument_family and instrument_source as classification targets
dataset = NSynth(
    "../data/nsynth/nsynth-train",
    transform=toFloat,
    blacklist_pattern=["string"],  # blacklist string instrument
    categorical_field_list=["instrument_family", "pitch"])
loader = data.DataLoader(dataset, batch_size=1, shuffle=True)


def expand(mat):
    expand_vec = np.expand_dims(mat[:, 125], axis=1)  # 取mat第125列插入新的维度
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


spec_list = []
pitch_list = []
IF_list = []
mel_spec_list = []
mel_IF_list = []

pitch_set = set()
count = 0
for samples, instrument_family, pitch, targets in loader:

    pitch = targets['pitch'].data.numpy()[0]

    if pitch < 24 or pitch > 84:
        #         print("pitch",pitch)
        continue

    sample = samples.data.numpy().squeeze() # squeeze掉batch的维（batch size为1）
    spec = librosa.stft(sample, n_fft=2048, hop_length=512)

    magnitude = np.log(np.abs(spec) + 1.0e-6)[:1024]
    #     print("magnitude Max",magnitude.max(),"magnitude Min",magnitude.min())
    angle = np.angle(spec) #因为要转为IF，所以这里不急着取到1024
    #     print("angle Max",angle.max(),"angle Min",angle.min())

    IF = phase_operation.instantaneous_frequency(angle, time_axis=1)[:1024]

    # 我猜这两个expand操作是为了下面转mel频率做铺垫
    magnitude = expand(magnitude)
    IF = expand(IF)
    logmelmag2, mel_p = spec_helper.specgrams_to_melspecgrams(magnitude, IF)

    #     pitch = targets['pitch'].data.numpy()[0]

    assert magnitude.shape == (1024, 128)
    assert IF.shape == (1024, 128)

    #     spec_list.append(magnitude)
    #     IF_list.append(IF)
    pitch_list.append(pitch)
    mel_spec_list.append(logmelmag2)
    mel_IF_list.append(mel_p)
    pitch_set.add(pitch)

    count += 1
    if count % 10000 == 0:
        print(count)

# train_data.create_dataset("Spec", data=spec_list)
# train_data.create_dataset("IF", data=IF_list)
train_data.create_dataset("pitch", data=pitch_list)
train_data.create_dataset("mel_Spec", data=mel_spec_list)
train_data.create_dataset("mel_IF", data=mel_IF_list)