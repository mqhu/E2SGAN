import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.EEGcls import EEGBlock, EEGSegment
from util.numpy_tools import make_dataset
from util.eeg_tools import Configuration
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
# from PGHI.preprocessing import AudioPreprocessor


conf = Configuration()

class EEGDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, patient_ls):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        phase_dir = 'train' if opt.phase in ['train', 'trainAsTest', 'perturbation'] else opt.phase
        dataIndex = np.load(os.path.join(opt.dataroot, 'dataIndex.npy'), allow_pickle=True).item()
        self.A_paths = []
        self.B_paths = []
        self.weights = []
        self.t2f = True if opt.domain == 'freq' else False

        for name in patient_ls:
            dataIdx = dataIndex[name]
            A_paths = make_dataset(os.path.join(opt.dataroot, 'A', phase_dir, name))
            B_paths = make_dataset(os.path.join(opt.dataroot, 'B', phase_dir, name))
            selected_APaths = []
            selected_BPaths = []
            if opt.phase == 'perturbation':
                topKresults = np.load('/public/home/xlwang/hmq/Infos/perturbation/top100FileNamesPSD.npy', allow_pickle=True).item()
                for file in topKresults[name]:
                    f_n = os.path.basename(file)
                    f_n = f_n.split('.')[0]
                    f_n = '_'.join(f_n.split('_')[:4]) + '.npy'
                    selected_APaths.append(os.path.join(opt.dataroot, 'A', 'train', name, f_n))
                    selected_BPaths.append(os.path.join(opt.dataroot, 'B', 'train', name, f_n))
                    # selected_APaths = [os.path.join(opt.dataroot, 'A', 'train', name, f_n)]
                    # selected_BPaths = [os.path.join(opt.dataroot, 'B', 'train', name, f_n)]
            elif phase_dir == 'train':
                for idx in dataIdx:
                    selected_APaths.append(A_paths[idx])
                    selected_BPaths.append(B_paths[idx])
            else:
                selected_APaths = A_paths
                selected_BPaths = B_paths
            self.A_paths += sorted(selected_APaths)
            self.B_paths += sorted(selected_BPaths)
            self.weights.append(len(selected_APaths))
        # self.B_paths = [sorted(make_dataset(dir_B, opt.max_dataset_size)) for dir_B in self.dir_B]  # 载入EEG channel的目录
        # assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        # if opt.pghi:
        #     self.preprocessor = AudioPreprocessor(sample_rate=conf.seeg_sf,
        #                       audio_length=conf.audio_length,
        #                       transform='pghi', hop_size=conf.seeg_hop, stft_channels=conf.seeg_n_fft).get_preprocessor()

        '''下面是带坐标距离的操作
        self.eeg_dist = {}
        ordered_eeg_chans = np.load(opt.ordered_eeg_chans)  # 这里需要提前保存每个病人的坐标差

        #  按照每个病人的每个seeg进行索引，存储seeg坐标与eeg坐标的坐标差
        for patient, mapping_ls in conf.seeg_mapping.items():
            self.eeg_dist[patient] = {}
            for seeg_name, mapped_eeg in mapping_ls.items():
                for eeg_name in ordered_eeg_chans:  # eeg顺序是固定好的
                    d = [x - y for x, y in zip(conf.eeg_pos[eeg_name], conf.eeg_pos[mapped_eeg])]
                    if seeg_name not in self.eeg_dist[patient].keys():
                        self.eeg_dist[patient][seeg_name] = []
                    self.eeg_dist[patient][seeg_name].append(d)  # 当前eeg到当前seeg的距离
        '''

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        # number, chan, patient = os.path.basename(A_path).split('_')  # 注意这里patient的结尾是.npy
        # B_path = os.path.join(self.dir_B, number, patient)
        B_path = self.B_paths[index]
        #AB = Image.open(AB_path).convert('RGB') #这里打开是PIL.Image格式，我的应该是ndarray
        # split AB image into A and B
        #w, h = AB.size
        #w2 = int(w / 2)
        A = EEGSegment(A_path)
        B = EEGSegment(B_path)
        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size())
        # if self.opt.pghi:
        #     A_transform = get_transform(self.opt, iseeg=False, t2f=self.t2f, preprocessor=self.preprocessor)
        #     B_transform = get_transform(self.opt, iseeg=True, t2f=self.t2f, preprocessor=self.preprocessor)
        # else:
        A_transform = get_transform(self.opt, iseeg=False, t2f=self.t2f)
        B_transform = get_transform(self.opt, iseeg=True, t2f=self.t2f)

        A = A_transform(A)  # 返回的是tensor
        B = B_transform(B)

        '''下面这是需要考虑坐标距离时的代码
        dist = []
        for rel_pos in self.eeg_dist[patient.split('.')[0]][chan]:
            d = np.linalg.norm(rel_pos, ord=1)
            if d == 0:
                d = .1
            dist.append(d ** -1)
        dist = torch.tensor(dist, dtype=torch.float)  # 距离的倒数
        '''

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}#, 'dist': dist}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return sum(self.weights)

    def get_class_weights(self):
        return self.weights


class EEGDatasetDataloader:

    def __init__(self, opt, patient_ls):

        self.opt = opt
        self.dataset = EEGDataset(opt, patient_ls)  # create a dataset given opt.dataset_mode and other options
        # if dataset_size is None:
        #     self.dataset_size = len(self.dataset)
        # else:
        #     self.dataset_size = dataset_size
        if opt.max_dataset_size == float('inf'):
            self.dataset_size = len(self.dataset)
        else:
            self.dataset_size = opt.max_dataset_size
        if opt.isTrain and 'eeggan' not in opt.model and len(opt.gpu_ids) > 0:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size,
                                                          sampler=self.train_sampler, num_workers=int(opt.num_threads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=opt.batch_size,
                    shuffle=not opt.serial_batches,
                    num_workers=int(opt.num_threads))

        # if opt.phase in ['train', 'test']:
        #     weights = self.dataset.get_class_weights()
        #     sampling_weights = []
        #     for weight in weights:
        #         sampling_weights += [weight for _ in range(weight)]
        #     sampling_weights = torch.tensor(sampling_weights, dtype=torch.float) ** -1
        #     sampler = WeightedRandomSampler(sampling_weights, self.dataset_size)
        #     self.dataloader = torch.utils.data.DataLoader(
        #         self.dataset,
        #         batch_size=opt.batch_size,
        #         # shuffle=not opt.serial_batches,
        #         shuffle=False,  # weightedsampling的时候要设置为False，免得label对应不上
        #         sampler=sampler,
        #         num_workers=int(opt.num_threads))
        # else:
        #     self.dataloader = torch.utils.data.DataLoader(
        #         self.dataset,
        #         batch_size=opt.batch_size,
        #         shuffle=not opt.serial_batches,
        #         num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.dataset_size:
                break
            yield data
