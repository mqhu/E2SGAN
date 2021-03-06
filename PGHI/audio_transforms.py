import numpy as np
import torch
from torch.nn.functional import interpolate
from librosa.core import stft, istft, magphase, resample


def complex_to_lin(x):
    return np.stack((np.real(x), np.imag(x)))

def lin_to_complex(x):
    assert len(x.shape) == 3, "Wrong shape"
    if type(x) is not np.ndarray:
        x = np.array(x)
    return x[0] + 1j * x[1]

def fade_out(x, percent=30.):
    """
        Applies fade out at the end of an audio vector

        x 
    """
    assert type(x) == np.ndarray, f"Fade_out: data type {type(x)} not {np.ndarray}"
    assert len(x.shape) == 1, f"Data has incompatible shape {x.shape}"

    fade_idx = int(x.shape[-1] * percent /100.)
    fade_curve = np.logspace(1, 0, fade_idx)
    fade_curve -= min(fade_curve)
    fade_curve /= max(fade_curve)
    x[-fade_idx:] *= fade_curve   
    return x

def fold_cqt(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    # For both mag and ph
    if x.size(0) == 2:

        mag = x[0]
        ph = x[1]
        hf = int(mag.size(0) / 2)

        mag1 = mag[:hf]
        mag2 = torch.from_numpy(np.flip(mag[hf:]).copy())
        ph1  = ph[:hf]
        ph2  = torch.from_numpy(np.flip(ph[hf:]).copy())
        return torch.stack([mag1, mag2, ph1, ph2], dim=0)
    # when only mag or ph
    elif x.size(0) == 1:
        split_idx = int(x.size(1) / 2)
        mag_l = x[0, :split_idx, :]
        mag_r = torch.Tensor(np.flip(x[0, split_idx:, :]).copy())

        return torch.stack([mag_l, mag_r], dim=0)

def unfold_cqt(x):
    # assert x.size(0) == 4, "fold_cqt: "
    if x.size(0) == 4:
        mag1 = x[0]
        mag2 = torch.from_numpy(np.flip(x[1]).copy())
        ph1  = x[2]
        ph2  = torch.from_numpy(np.flip(x[3]).copy())
        mag  = torch.cat([mag1, mag2], dim=0)
        ph   = torch.cat([ph1, ph2], dim=0)
        return torch.stack([mag, ph], dim=0)
    elif x.size(0) == 2:
        mag1 = x[0]
        mag2 = torch.from_numpy(np.flip(x[1]).copy())
        return torch.cat([mag1, mag2], dim=0).unsqueeze(0)

def norm_audio(x):
    if max(abs(x)) != 0:
        return x/max(abs(x))
    else:
        return x

def mag_phase_angle(x):
    mag, ph = magphase(x)
    ph = np.angle(ph)
    out = np.stack([mag, ph])
    return out

def mag_to_complex(x):
    mag = x[0]
    ph = x[1]
    return np.array(mag) * np.exp(1.j * np.array(ph))

def safe_log(x):
    return torch.log(x + 1e-10)

def safe_log_spec(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if x.size(0) == 2:
        mag = x[0]
        ph = x[1]
        mlog = safe_log(mag)
        return torch.stack([mlog, ph], dim=0)
    elif x.size(0) == 1:
        return safe_log(x)

def safe_exp_spec(x):

    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if x.size(0) == 2:
        mag = x[0]
        ph = x[1]
        elog = safe_exp(mag)
        return torch.stack([elog, ph], dim=0)
    elif x.size(0) == 1:
        return safe_exp(x[0]).unsqueeze(0)

def safe_exp(x):
    return torch.exp(x) - 1e-10

class ResampleWrapper():
    def __init__(self, orig_sr, target_sr):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
    def __call__(self, audio):
        return torch.Tensor(resample(y=audio.numpy(), target_sr=self.target_sr, orig_sr=self.orig_sr))
    

class RemoveDC():
    def __init__(self, fdim=-2):
    # def __init__(self, fdim=1):
        self.fdim=fdim

    def __call__(self, spectrum):
        assert len(spectrum.shape) == 3, \
            f"Spectrum shape {spectrum.shape} not valid"
        assert spectrum.shape[-2] % 2 != 0, "Is dim 2 freq?"
        # self.fdim = np.argmax(spectrum.size())
        return spectrum[:, :-1, :]

class AddDC():
    def __init__(self, fdim=-2):
        self.fdim = fdim
    def __call__(self, x):
        if type(x) is not np.ndarray: 
            x = x.numpy()
        return np.concatenate(
            [x, x[:, -1:, :]], axis=-2)

class ResizeWrapper():
    def __init__(self, new_size):
        self.size = new_size
    def __call__(self, image):
        assert np.argmax(self.size) == np.argmax(image.shape[-2:]), \
            f"Resize dimensions mismatch, Target shape {self.size} \
                != image shape {image.shape}"
        if type(image) is not np.ndarray:
            image = image.numpy()
        out = interpolate(torch.from_numpy(image).unsqueeze(0), size=self.size).squeeze(0)
        return out

class LibrosaIstftWrapper():
    def __init__(self, hop_length, win_length):
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, x):
        # x = torch.stack([x[0].t(), x[1].t()], dim=0)
        x = mag_to_complex(x)
        return istft(x,
                     hop_length=self.hop_length,
                     win_length=self.win_length) 

class LibrosaStftWrapper():
    def __init__(self, n_fft=4096, ws=2048, hop=1024):
        self.n_fft=n_fft
        self.ws=ws
        self.hop=hop

    def __call__(self, audio):
        spec = stft(audio.numpy().reshape(-1,),
                    hop_length=self.hop,
                    win_length=self.ws,
                    n_fft=self.n_fft)
        mag, ph = magphase(spec)
        mag = torch.Tensor(mag)
        ph = np.angle(ph)
        ph = torch.Tensor(ph)
        out = torch.stack((mag, ph), dim=0)
        return out


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            # print('transform:', t)
            # print('before:')
            # print(audio)
            audio = t(audio)
            # print('after:')
            # print(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def remove_ph(spectrum):
  return spectrum[0:1]

def phase_diff(ph):
    return torch.Tensor(ph[:, 1:] - ph[:, :-1])

def inv_instantanteous_freq(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    ifreq_inv = torch.stack([x[0], torch.cumsum(x[1] * np.pi, dim=1)])
    
    return ifreq_inv

def instantaneous_freq(specgrams):
    if specgrams.shape[0] != 2:
      ph = specgrams
    else:
      mag = specgrams[0]
      ph = specgrams[1]

    uph = np.unwrap(ph, axis=1)
    uph_diff = np.diff(uph, axis=1)
    ifreq = np.concatenate([ph[:, :1], uph_diff], axis=1)

    if specgrams.shape[0] == 2:
      return np.stack([mag, ifreq/np.pi])
    else:
      return ifreq