import torch

from .audio_transforms import complex_to_lin, lin_to_complex, \
    RemoveDC, Compose, safe_log, mag_to_complex, \
    AddDC, safe_exp, safe_log_spec, safe_exp_spec, mag_phase_angle, norm_audio, fold_cqt, unfold_cqt, fade_out, \
    instantaneous_freq, inv_instantanteous_freq, instantaneous_freq

from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale
from librosa.core import magphase
import numpy as np
from torch.nn.modules.padding import ConstantPad2d
import librosa
import pdb
from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.stft import GaussTF, GaussTruncTF
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram

# TO-DO: add function to get the output of the pipelin
# on intermediate positions

class DataPreprocessor(object):
    """
        This class manages all datasets and given
        the config for each of them it returns
        the corresponding DataLoader object and
        the pre-processing transforms.
    """
    # define available audio transforms
    AUDIO_TRANSFORMS = ["waveform", "stft", "mel", "cqt", "cq_nsgt", "specgrams", "mfcc", "pghi"]

    def __init__(self,
                 transform=None,
                 preprocess=False,
                 **kwargs):
        """
            Creates a data manager

            @arg

        """

        self.pre_pipeline = []
        self.post_pipeline = []
        self.set_atts(**kwargs)
        self.init_transform_pipeline(transform)

    def set_atts(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def init_transform_pipeline(self, transform):
        raise NotImplementedError

    def get_preprocessor(self, compose=True):
        if not compose: return self.pre_pipeline
        return Compose(self.pre_pipeline)

    def get_postprocessor(self, compose=True):
        if not compose: return self.post_pipeline
        return Compose(self.post_pipeline)



class AudioPreprocessor(DataPreprocessor):
    def __init__(self,
                 sample_rate=16000,
                 audio_length=16000,
                 transform='waveform',
                 **kargs):
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        DataPreprocessor.__init__(self, transform=transform, **kargs)


    

    def set_per_batch_transform(self, transform):
        self.loader.set_transform(transform)

    def init_transform_pipeline(self, transform):
        """
            Function that initializes the transformation pipeline

            Args:
                transform (str): name of the transformation
        """

        # Domain specific transforms
        assert transform in self.AUDIO_TRANSFORMS, \
            f"Transform '{transform}' not in {self.AUDIO_TRANSFORMS}"
        
        print(f"Configurign {transform} transform...")
        self.transform = transform
        {   
            "waveform":  self.build_waveform_pipeline,
            "stft":      self.build_stft_pipeline,
            "specgrams": self.build_specgrams_pipeline,
            "mel":       self.build_mel_pipeline,
            "cqt":       self.build_cqt_pipeline,
            "cq_nsgt":   self.build_cqt_nsgt_pipeline,
            "mfcc":      self.build_mfcc_pipeline,
            "pghi":      self.build_pghi_pipeline
        }[self.transform]()

    def build_waveform_pipeline(self): 
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self.output_shape = (1, 1, self.audio_length)

        def in_reshape(x):
            return x.reshape(self.audio_length)
        def out_reshape(x):
            return x.reshape(self.output_shape)

        self.post_pipeline.insert(0, in_reshape)
        self.pre_pipeline.append(out_reshape)

    def build_stft_pipeline(self):
        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self._add_stft()
        self._complex_to_lin()
        self._add_rm_dc()
        self.output_shape = (2, self.n_bins, self.n_frames)

    def build_specgrams_pipeline(self):
        self._init_stft_params()
        # self._add_audio_loader()
        self._add_signal_zeropadding()
        # self._add_fade_out()
        # self._add_norm()
        self._add_stft()
        self._add_mag_phase()
        self._add_rm_dc()
        self._add_log_mag()
        self._add_ifreq()
        # TO-DO: add rm magnitude
        # self.output_shape = (2, self.n_bins, self.n_frames)

    def build_mel_pipeline(self):
        def mel(x):
            return librosa.feature.melspectrogram(
                x.reshape(-1), 
                sr=self.sample_rate,
                n_fft=getattr(self, 'fft_size', 2048),
                hop_length=self.hop_size,
                win_length=getattr(self, 'win_size', 1024),
                n_mels=getattr(self, 'n_mels', 128)
            )
        def imel(x):
            return librosa.feature.inverse.mel_to_audio(
                x.squeeze(0),
                sr=self.sample_rate,
                n_iter=getattr(self, 'gl_n_iter', 100),
                n_fft=getattr(self, 'fft_size', 2048), 
                hop_length=self.hop_size,
                win_length=getattr(self, 'win_size', 1024))
        def reshape(x):
            return x.reshape(self.output_shape)
        def to_numpy(x):
            if type(x) == np.ndarray:
                return x
            return x.numpy()

        self._init_stft_params()
        
        self.output_shape = (1, getattr(self, 'n_mels', 128), self.n_frames) 

        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self.pre_pipeline.append(mel)
        self.post_pipeline.insert(0, imel)
        self.pre_pipeline.append(reshape)
        self.post_pipeline.insert(0, reshape)

        # self._add_log_mag()
        self.post_pipeline.insert(0, to_numpy)
        

    def build_mfcc_pipeline(self):
        def reshape(x):
            return x.reshape(self.output_shape)
        def to_numpy(x):
            if type(x) == np.ndarray:
                return x
            return x.numpy()

        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        mfcc = \
            lambda x : librosa.feature.mfcc(
                y=x, 
                sr=self.sample_rate,
                n_fft=getattr(self, 'fft_size', 2048),
                n_mels=getattr(self, 'n_mel', 128),
                hop_length=self.hop_size,
                win_length=getattr(self, 'win_size', 1024),
                S=None, 
                n_mfcc=getattr(self, 'n_mfcc', 20), 
                dct_type=2, 
                norm='ortho', 
                lifter=0)
        imfcc = \
            lambda x: librosa.feature.inverse.mfcc_to_audio(
                x.squeeze(0),
                n_mels=getattr(self, 'n_mel', 128),
                sr=self.sample_rate,
                n_iter=getattr(self, 'gl_n_iter', 100),
                n_fft=getattr(self, 'fft_size', 2048),
                win_length=getattr(self, 'win_size', 1024),
                hop_length=self.hop_size,
                dct_type=2, 
                norm='ortho', 
                ref=1.0)

        self.pre_pipeline.append(mfcc)
        self.post_pipeline.insert(0, imfcc)
    
        self.output_shape = (1, getattr(self, 'n_mfcc', 128), self.n_frames)
        
        self.pre_pipeline.append(reshape)
        self.post_pipeline.insert(0, reshape)
        self.post_pipeline.insert(0, to_numpy)
        
        
    def build_cqt_pipeline(self):
        def reshape(x):
            return x.reshape(self.output_shape)

        def cqt(x):
            return librosa.core.cqt(x,
                               sr=self.sample_rate,
                               hop_length=self.hop_size,
                               n_bins=getattr(self, 'n_cqt', 84),
                               bins_per_octave=getattr(self, 'bins_per_octave', 12))
        def icqt(x):
            return librosa.core.icqt(x,
                                     sr=self.sample_rate,
                                     hop_length=self.hop_size)
        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()

        self.pre_pipeline.append(cqt)
        self.post_pipeline.insert(0, icqt)
        self._add_mag_phase()
        self._add_log_mag()
        self._add_ifreq()

        self.output_shape = (2, getattr(self, 'n_cqt', 84), self.n_frames)
        self.pre_pipeline.append(reshape)
        self.post_pipeline.insert(0, reshape)

    def build_cqt_nsgt_pipeline(self):
        print("")
        print("Configuring cqt_NSGT pipeline...")

        scales  = {'log':LogScale,
                   'lin':LinScale,
                   'mel':MelScale,
                   'oct':OctScale}
        nsgt_scale = scales[getattr(self, 'nsgt_scale', 'log')]
        nsgt_scale = nsgt_scale(getattr(self, 'fmin', 20),
                                getattr(self, 'fmax', self.sample_rate / 2),
                                getattr(self, 'n_bins', 96))
        nsgt = NSGT(nsgt_scale,
                    self.sample_rate, 
                    self.audio_length, 
                    real=getattr(self, 'real', False), 
                    matrixform=getattr(self, 'matrix_form', True), 
                    reducedform=getattr(self, 'reduced_form', False))
        self.n_bins = len(nsgt.wins)
        self.n_frames = nsgt.ncoefs

        self.output_shape = (2, int(self.n_bins/2), int(self.n_frames))
        ######### TRANSFORMS #########
        reshape = lambda x: x.reshape(-1,)

        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self.pre_pipeline.extend([reshape, nsgt.forward])
        self.post_pipeline.insert(0, nsgt.backward)
        self._add_mag_phase()
        self._add_log_mag()
        self._add_ifreq()
        # Add folded cqt
        if getattr(self, 'fold_cqt', False):
            self.pre_pipeline.append(fold_cqt)
            self.post_pipeline.insert(0, unfold_cqt)
            self.output_shape = (4, int(self.n_bins/2), int(self.n_frames))

    def build_pghi_pipeline(self):
        self._pghi_init_stft_params()
        # self._pghi_add_audio_loader()
        self._add_signal_zeropadding()
        self._pghi_add_stft()
        # self.output_shape = (2, self.n_bins, self.n_frames)


    def _add_audio_loader(self):
        def loader(x):
            return librosa.core.load(
            x,
            sr=self.sample_rate,
            mono=True,
            offset=0.0,
            duration=self.audio_length / self.sample_rate,
            dtype=np.float32,
            res_type='kaiser_best')[0]  # We index 0 to remove the sample rate
        self.pre_pipeline.append(loader)

    def _pghi_add_audio_loader(self):
        def pghi_loader(x):
            y, sr = load_signal(x)
            y = preprocess_signal(y)
            return y
        self.pre_pipeline.append(pghi_loader)

    def _add_fade_out(self):
                
        # Common transforms
        if getattr(self, 'fade_out', False):
            self.pre_pipeline.append(fade_out)

    def _add_norm(self):
        if getattr(self, 'normalization', False):
            self.post_pipeline.insert(0, norm_audio)
    def _complex_to_lin(self):
            self.pre_pipeline.append(complex_to_lin) 
            self.post_pipeline.insert(0, lin_to_complex)

    def _add_signal_zeropadding(self):
        def zeropad(signal):
            if len(signal) < self.audio_length:
                return np.append(
                    signal, 
                    np.zeros(self.audio_length - len(signal))
                )
            else:
                signal = signal[0:self.audio_length]
                return signal
        self.pre_pipeline.append(zeropad)


    def _init_stft_params(self):
        if not hasattr(self, 'hop_size'):
            self.hop_size = int(getattr(self, 'win_size', 1024) / 2)
        if hasattr(self, 'n_frames') and not hasattr(self, 'audio_length'):
            self.audio_length = self.n_frames * self.hop_size - 1  # we substract one so we get exactly self.n_frames

        self.n_bins = int(getattr(self, 'fft_size', 2048) / 2)

    def _pghi_init_stft_params(self):
        if not hasattr(self, 'hop_size'):
            self.hop_size = int(getattr(self, 'stft_channels', 512) / 2)
        if hasattr(self, 'n_frames'):
            self.audio_length = self.n_frames * self.hop_size
        self.n_bins = int(getattr(self, 'stft_channels', 512) / 2)+1

    def _add_stft(self):
        def stft(x):
            return librosa.core.stft(
                    x,
                    hop_length=getattr(self, 'hop_size', 512),
                    win_length=getattr(self, 'win_size', 1024),
                    n_fft=getattr(self, 'fft_size', 1024))
        def istft(x):
            return librosa.core.istft(
                    x,
                    hop_length=getattr(self, 'hop_size', 512),
                    win_length=getattr(self, 'win_size', 1024))
        self.pre_pipeline.append(stft)
        self.post_pipeline.insert(0, istft)

    def _pghi_add_stft(self):
        def pghi_stft(x):
            use_truncated_window = True
            if use_truncated_window:
                stft_system = GaussTruncTF(hop_size=getattr(self, 'hop_size', 256), stft_channels=getattr(self, 'stft_channels', 512))
            else:
                stft_system = GaussTF(hop_size=getattr(self, 'hop_size', 256), stft_channels=getattr(self, 'stft_channels', 512))
            Y = stft_system.spectrogram(x, normalize=False)
            log_Y = log_spectrogram(Y)
            return np.expand_dims(log_Y,axis=0)
            #return log_Y

        def pghi_istft(x):
            use_truncated_window = True
            if use_truncated_window:
                stft_system = GaussTruncTF(hop_size=getattr(self, 'hop_size', 256), stft_channels=getattr(self, 'stft_channels', 512))
            else:
                stft_system = GaussTF(hop_size=getattr(self, 'hop_size', 256), stft_channels=getattr(self, 'stft_channels', 512))

            if type(x) is not np.ndarray:
                x = np.squeeze(x.numpy(),axis=0)
            new_Y = inv_log_spectrogram(x)
            new_y = stft_system.invert_spectrogram(new_Y)
            return new_y

        self.pre_pipeline.append(pghi_stft)
        self.post_pipeline.insert(0, pghi_istft)

    def _add_mag_phase(self):
        # Add complex to mag/ph transform
        mag_ph = lambda x: mag_phase_angle(x)
        i_mag_ph = lambda x: mag_to_complex(x)
        self.pre_pipeline.append(mag_ph)
        self.post_pipeline.insert(0 , i_mag_ph)

    def _add_rm_dc(self):
        if getattr(self, 'rm_dc', True):
            # Remove DC transforms
            self.pre_pipeline.append(RemoveDC())
            self.post_pipeline.insert(0, AddDC())

    def _add_log_mag(self):
        # Log magnitude
        if getattr(self, 'log_mag', True):
            self.pre_pipeline.append(safe_log_spec)
            self.post_pipeline.insert(0, safe_exp_spec)

    def _add_ifreq(self):
        if getattr(self, 'ifreq', False):
            self.pre_pipeline.append(instantaneous_freq)
            self.post_pipeline.insert(0, inv_instantanteous_freq)

    def get_output_shape(self):
        return list(self.output_shape)

    def get_post_processor(self, insert_transform=None):
        if insert_transform is None:
            return Compose(self.post_pipeline)
        return Compose([insert_transform] + self.post_pipeline)

