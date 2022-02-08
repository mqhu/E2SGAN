import math
import os
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import pdb
from tqdm import tqdm

from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg

from data.EEGcls import EEGSegment
from util.eeg_tools import IF_to_eeg
import util.numpy_tools as nutil
import data.base_dataset as bd
from gansynth.normalizer import DataNormalizer
import mne
import scipy


class EEG(torch.utils.data.Dataset):

    def __init__(self, dir, isreal, isIF=False, normalizer=None, transforms=None):

        self.dir = sorted(nutil.make_dataset(dir))
        self.isreal = isreal
        self.isIF = isIF
        self.normalizer = normalizer
        self.tranforms = transforms

    def __getitem__(self, index):

        if self.isreal:
            path = self.dir[index]
        else:
            path = self.dir[index * 3]
        segment = EEGSegment(path, isIF=self.isIF, normalizer=self.normalizer)
        segment = self.tranforms(segment)

        return segment

    def __len__(self):

        if self.isreal:
            return len(self.dir)
        else:
            return len(self.dir) // 3


def get_max_min_normalize_params(dataloader, save_name):  # 我觉得应该在训练集上跑（原始eeg）

    max_v = float("-inf")
    min_v = float("inf")
    itr = 0
    for idx, data in enumerate(dataloader):
        data = data.numpy()
        max_v = data.max() if data.max() > max_v else max_v
        min_v = data.min() if data.min() < min_v else min_v
        itr += 1

    print("Iterations: " + str(itr))
    d = {'max_v': max_v, 'min_v': min_v}
    print(d)
    np.save(save_name, d)


def __max_min_normalize(img):  # 针对EEGSegment

    data = img.get_data()
    max_v = 0.0029741877
    min_v = -0.0030734031
    data = (data - min_v) / (max_v - min_v)
    img.set_data(data)
    return img


def calculate_means_and_var(dataloader, save_name): # 这里是对原始EEGSegment的batch进行计算

    means = 0
    total = 0
    flag = True
    data_shape = None
    for idx, data in enumerate(dataloader):
        data = data.numpy()
        if flag:
            data_shape = data.shape
        means += np.sum(data)
        total += data.shape[0]

    total = total * data_shape[1] * data_shape[2] # 图片数*像素数
    means /= total

    tmp = 0
    for idx, data in enumerate(dataloader):
        data = data.numpy()
        tmp += np.sum((data - means) ** 2)

    std = np.sqrt(tmp / total)
    d = {'means': means, 'std': std}
    print(d)
    np.save(save_name, d)


def get_transforms(model):
    transform = [transforms.Lambda(lambda img: __max_min_normalize(img)),
                 transforms.Lambda(lambda img: bd.__square_size(img, 224)),
                 transforms.Lambda(lambda img: bd.__1ch_to_3ch(img)),
                 transforms.Lambda(lambda img: bd.__np_to_tensor(img))]
    if model.find('inception') >= 0:
        transform[1] = transforms.Lambda(lambda img: bd.__square_size(img, 299))

    return transforms.Compose(transform)

def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)


def make_dataset(dataset, dataroot, imageSize):
    """
    :param dataset: must be in 'cifar10 | lsun | imagenet | folder | lfw | fake'
    :return: pytorch dataset for DataLoader to utilize
    """
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(imageSize),
                                       transforms.CenterCrop(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'lsun':
        dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset == 'celeba':
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(138),
                                       transforms.Resize(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        raise Exception('--dataset must be in cifar10 | lsun | imagenet | folder | lfw | fake')
    assert dataset
    return dataset


def sampleFake(netG, nz, sampleSize, batchSize, saveFolder):
    print('sampling fake images ...')
    saveFolder = saveFolder + '0/'

    try:
        os.makedirs(saveFolder)
    except OSError:
        pass

    noise = torch.FloatTensor(batchSize, nz, 1, 1).cuda()
    iter = 0
    for i in range(0, 1 + sampleSize // batchSize):
        noise.data.normal_(0, 1)
        fake = netG(noise)
        for j in range(0, len(fake.data)):
            if iter < sampleSize:
                vutils.save_image(fake.data[j].mul(0.5).add(
                    0.5), saveFolder + giveName(iter) + ".png")
            iter += 1
            if iter >= sampleSize:
                break


def sampleTrue(dataset, imageSize, dataroot, sampleSize, batchSize, saveFolder, workers=4):
    # 大概是因为原数据集图片太多了，所以这里要采样一点，我觉得我不需要
    print('sampling real images ...')
    saveFolder = saveFolder + '0/'

    dataset = make_dataset(dataset, dataroot, imageSize)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batchSize, num_workers=int(workers))

    if not os.path.exists(saveFolder):
        try:
            os.makedirs(saveFolder)
        except OSError:
            pass

    iter = 0
    for i, data in enumerate(dataloader, 0):
        img, _ = data
        for j in range(0, len(img)):

            vutils.save_image(img[j].mul(0.5).add(
                0.5), saveFolder + giveName(iter) + ".png")
            iter += 1
            if iter >= sampleSize:
                break
        if iter >= sampleSize:
            break


class ConvNetFeatureSaver(object):
    def __init__(self, model='resnet34', workers=4, batchSize=64):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.batch_size = batchSize
        self.workers = workers
        self.trans = get_transforms(model)
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
        else:
            raise NotImplementedError

    def save(self, imgFolder, isreal, isIF=False, normalizer=None, save2disk=False):
        #dataset = dset.ImageFolder(root=imgFolder, transform=self.trans)
        dataset = EEG(imgFolder, isreal=isreal, isIF=isIF, normalizer=normalizer, transforms=self.trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.workers)
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []
        for img in tqdm(dataloader):
            with torch.no_grad():
                input = img.cuda()
                if self.model == 'vgg' or self.model == 'vgg16':
                    fconv = self.vgg.features(input).view(input.size(0), -1)
                    flogit = self.vgg.classifier(fconv)
                    # flogit = self.vgg.logitifier(fconv)
                elif self.model.find('resnet') >= 0:
                    fconv = self.resnet_feature(
                        input).mean(3).mean(2)
                    flogit = self.resnet.fc(fconv)
                elif self.model == 'inception' or self.model == 'inception_v3':
                    fconv = self.inception_feature(
                        input).mean(3).mean(2)
                    flogit = self.inception.fc(fconv)
                else:
                    raise NotImplementedError
                fsmax = F.softmax(flogit)
                feature_pixl.append(img)
                feature_conv.append(fconv.data.cpu())
                feature_logit.append(flogit.data.cpu())
                feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0).to('cpu')
        feature_conv = torch.cat(feature_conv, 0).to('cpu')
        feature_logit = torch.cat(feature_logit, 0).to('cpu')
        feature_smax = torch.cat(feature_smax, 0).to('cpu')

        if save2disk:
            torch.save(feature_conv, os.path.join(
                imgFolder, 'feature_pixl.pth'))
            torch.save(feature_conv, os.path.join(
                imgFolder, 'feature_conv.pth'))
            torch.save(feature_logit, os.path.join(
                imgFolder, 'feature_logit.pth'))
            torch.save(feature_smax, os.path.join(
                imgFolder, 'feature_smax.pth'))

        return feature_pixl, feature_conv, feature_logit, feature_smax


def distance(X, Y, sqrt):
    X = X.cpu()
    Y = Y.cpu()
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) - 
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))  # idx大概就是每个样本对应距离最近的位置
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])  # 根据位置返回标签
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)  # 预测为正样本的结果中有多少是正确的
    s.recall = s.tp / (s.tp + s.fn + 1e-10)  # 正样本中有多少被识别正确
    s.acc_real = s.tp / (s.tp + s.fn)
    s.acc_fake = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd


def entropy_score(X, Y, epsilons):
    Mxy = distance(X, Y, False)
    scores = []
    for epsilon in epsilons:
        scores.append(ent(Mxy.t(), epsilon))

    return scores


def ent(M, epsilon):
    n0 = M.size(0)
    n1 = M.size(1)
    neighbors = M.lt(epsilon).float()
    sums = neighbors.sum(0).repeat(n0, 1)
    sums[sums.eq(0)] = 1
    neighbors = neighbors.div(sums)
    probs = neighbors.sum(1) / n1
    rem = 1 - probs.sum()
    if rem < 0:
        rem = 0
    probs = torch.cat((probs, rem*torch.ones(1)), 0)
    e = {}
    e['probs'] = probs
    probs = probs[probs.gt(0)]
    e['ent'] = -probs.mul(probs.log()).sum()

    return e



eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score

def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


def integral(x, y):  # 矩形近似求积分

    sum = 0
    for i in range(len(x) - 1):
        sum += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.
    return sum


def freq_band_psd(signals):  # 传入的是ndarray
    '''返回形状: (n_chans, n_freq_bands)， 这里n_freq_bands为12(64//5)'''
    sfreq = 128.
    psds, freqs = mne.time_frequency.psd_array_multitaper(signals, sfreq, fmax=sfreq/2)
    band_w = 5  # 每5赫兹一个频段进行统计

    freq_end = sfreq / 2
    result = []
    for ch in psds:
        p = 0
        t = []
        freq_cnt = 0
        for i in range(len(freqs)):
            if freqs[i] > freq_end:
                break
            if freqs[i] > freq_cnt + band_w:
                t.append(integral(freqs[p: i - 1], ch[p: i - 1]))
                freq_cnt += band_w
                p = i
        result.append(t)

    return np.asarray(result)


def get_correlation_mat(X_dataset, Y_dataset, n_seeg, order, n_freq_band=12, is_fake=False, normalizer=None):
    '''一般X是seeg，Y是eeg'''
    X_f = []
    Y_f = []

    flag = True
    for f_n in X_dataset:
        signal = np.load(f_n)
        X_freq_psd = freq_band_psd(signal)
        if flag:
            flag = False
            X_f = [[] for _ in range(X_freq_psd.shape[0] * X_freq_psd.shape[1])]
        cnt = 0
        for ch in X_freq_psd:
            for psd in ch:
                X_f[cnt].append(psd)
                cnt += 1
    X_f = np.asarray(X_f)

    flag = True
    for f_n in Y_dataset:
        signal = np.load(f_n)
        if is_fake:
            signal = IF_to_eeg(signal, normalizer)
        Y_freq_psd = freq_band_psd(signal)
        if flag:
            flag = False
            Y_f = [[] for _ in range(Y_freq_psd.shape[0] * Y_freq_psd.shape[1])]
        cnt = 0
        for ch in Y_freq_psd:
            for psd in ch:
                Y_f[cnt].append(psd)
                cnt += 1
    Y_f = np.asarray(Y_f)

    #pca = PCA(n_components=10)
    #reduced_X = pca.fit_transform(X_f)
    #reduced_Y = pca.fit_transform(Y_f)

    #corr_mat = np.corrcoef(X_f, Y_f)
    #xcorr = corr_mat[: n_seeg * n_freq_band, n_seeg * n_freq_band:]  # 截取互相关系数
    xcorr = []

    for pair in order:
        eeg_idx = pair[0]
        seeg_idx = pair[1]
        for i in range(12):
            for j in range(12):
                corr = np.corrcoef(X_f[seeg_idx * 12 + i], Y_f[eeg_idx * 12 + j])
                xcorr.append(corr[0][1])

    return xcorr#, pca.explained_variance_ratio_


def test_correlation_significance(A,  n_A_samples):  # t检验相关系数显著性

    alpha = 0.05

    '''for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i][j]
            t = abs(a) * ((n_A_samples - 2) / (1 - a ** 2)) ** 0.5  # t检验统计量
            t_critical = scipy.stats.t.isf(alpha / 2., df=(n_A_samples - 2))
            if t < t_critical:  # 不具有统计显著性的话就标记
                A[i][j] = 2'''

    for i in range(len(A)):
        a = A[i]
        t = abs(a) * ((n_A_samples - 2) / (1 - a ** 2)) ** 0.5  # t检验统计量
        t_critical = scipy.stats.t.isf(alpha / 2., df=(n_A_samples - 2))
        if t < t_critical:  # 不具有统计显著性的话就标记
            A[i] = 2

    return A


def compare_correlation_mat(A, B, n_A_samples, n_B_samples):
    '''
    矩阵A和B的行顺序是先seeg后eeg
    '''
    total = 0
    same = 0

    A = test_correlation_significance(list(A), n_A_samples)
    B = test_correlation_significance(list(B), n_B_samples)
    print(A)
    print(B)
    print(len(A[A == 2]))
    print(len(B[B == 2]))
    print(A.shape)
    print(B.shape)
    a_cnt = 0
    b_cnt = 0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = abs(A[i][j])
            b = abs(B[i][j])
            if a == 2:  # 不具有统计显著性的话就跳过
                continue
            else:
                total += 1
            if a > 0.5:
                a_cnt += 1
            if b > 0.5:
                b_cnt += 1
            '''费雪变换'''
            z1 = 0.5 * math.log((1.0 + a) / (1.0 - a))
            z2 = 0.5 * math.log((1.0 + b) / (1.0 - b))
            '''z检验判断两个相关系数是否有统计显著性差异'''
            z_observed = abs(z1 - z2) / (1.0 / (n_A_samples - 3) + 1.0 / (n_B_samples - 3)) ** 0.5
            if z_observed < 1.96:  # 1.96是显著性0.05时的正态分布的值
                same += 1
    score = same / total
    print("total:", total)
    print("same:", same)
    print('A over 0.5:', a_cnt)
    print('B over 0.5:', b_cnt)

    return score


def compare_correlation_array(A, B, n_A_samples, n_B_samples):
    '''
    矩阵A和B的行顺序是先seeg后eeg
    A是real, B是fake
    '''
    total = 0
    same = 0

    A = test_correlation_significance(list(A), n_A_samples)
    B = test_correlation_significance(list(B), n_B_samples)
    #print(A)
    #print(B)
    #print(A.count(2))
    #print(B.count(2))
    #print(len(A))
    #print(len(B))
    a_cnt = 0
    b_cnt = 0
    idx = [0 for _ in range(len(A))]

    for i in range(len(A)):
        a = abs(A[i])
        b = abs(B[i])
        '''if a != 2:
            total += 1
        if a == 2 or b == 2:  # 不具有统计显著性的话就跳过
            continue'''
        if a == 2 or b == 2:
            continue
        else:
            total += 1
        if a > 0.5:
            a_cnt += 1
        if b > 0.5:
            b_cnt += 1
        '''费雪变换'''
        try:
            z1 = 0.5 * math.log((1.0 + a) / (1.0 - a))
            z2 = 0.5 * math.log((1.0 + b) / (1.0 - b))
        except ValueError:
            print('a=', a)
            print('b=', b)
        '''z检验判断两个相关系数是否有统计显著性差异'''
        z_observed = abs(z1 - z2) / (1.0 / (n_A_samples - 3) + 1.0 / (n_B_samples - 3)) ** 0.5
        if z_observed < 1.96:  # 1.96是显著性0.05时的正态分布的值
            same += 1
            idx[i] = 1
    score = same / total
    print("total:", total)
    print("same:", same)
    print("score:", score)
    #print('A over 0.5:', a_cnt)
    #print('B over 0.5:', b_cnt)

    return score, A.count(2), B.count(2), a_cnt, b_cnt, idx


class Score:
    emd = 0
    mmd = 0
    knn = None


def compute_score(real, fake, k=1, sigma=1, sqrt=True):

    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return s


def compute_score_raw(batchSize, saveFolder_r, saveFolder_f, save_name, normalizer=None, conv_model='resnet34', workers=4):
    '''saveFolder_r和saveFolder_f分别是真图和假图的目录'''

    convnet_feature_saver = ConvNetFeatureSaver(model=conv_model,
                                                batchSize=batchSize, workers=workers)
    feature_r = convnet_feature_saver.save(saveFolder_r, isreal=True, isIF=False, normalizer=None)
    feature_f = convnet_feature_saver.save(saveFolder_f, isreal=False, isIF=True, normalizer=normalizer)

    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(36)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i].to(device), feature_r[i].to(device), False)
        Mxy = distance(feature_r[i].to(device), feature_f[i].to(device), False)
        Myy = distance(feature_f[i].to(device), feature_f[i].to(device), False)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_real, tmp.acc_fake, tmp.precision, tmp.recall

    score[28] = inception_score(feature_f[3])
    score[29] = mode_score(feature_r[3], feature_f[3])
    score[30] = fid(feature_r[1], feature_f[1])  # 有人说应该在conv层计算，即feature_r/f[1]

    '''seeg_ds = nutil.make_dataset("/home/cbd109-3/Users/data/hmq/GANDatasets/LK_rest/A/test/")
    eeg_ds_r = nutil.make_dataset(saveFolder_r)
    eeg_ds_ff = nutil.make_dataset(saveFolder_f)
    eeg_ds_f = []
    for f_n in eeg_ds_ff:
        if f_n.find('fake') > 0:
            eeg_ds_f.append(f_n)
    eeg_seeg_pairs = np.load("/home/cbd109/Users/hmq/LK_info/eeg_seeg_pairs_all_eeg.npy", allow_pickle=True).item()['pairs']
    inclusive_eeg = []
    for ch in eeg_ch:
        if ch not in exclusions:
            inclusive_eeg.append(ch)
    order = []
    for pair in eeg_seeg_pairs:
        try:
            eeg_idx = inclusive_eeg.index(pair[0])
            seeg_idx = seeg_ch.index(pair[1])
        except:
            continue
        order.append((eeg_idx, seeg_idx))
    corr_r = get_correlation_mat(seeg_ds, eeg_ds_r, 130, order)
    corr_f = get_correlation_mat(seeg_ds, eeg_ds_f, 130, order, is_fake=True, normalizer=normalizer)
    corr_score = compare_correlation_array(corr_r, corr_f, 120, 120)
    score[31] = corr_score[0]
    score[32] = corr_score[1]
    score[33] = corr_score[2]
    score[34] = corr_score[3]
    score[35] = corr_score[4]'''
    for i in range(31, 36):
        score[i] = 0
    #np.save(save_name, score)
    return score  # score是ndarray


def print_score(score):

    for i in range(4):
        feature = ''
        if i == 0:
            feature = 'Raw image'
        elif i == 1:
            feature = 'Conve'
        elif i == 2:
            feature = 'Fc'
        else:
            feature = 'Softmax'
        print(feature + ':')
        print("Wasserstein distance:" + str(score[i * 7]), end=', ')
        print("MMD:" + str(score[i * 7 + 1]), end=', ')
        print("1-NN accuracy:" + str(score[i * 7 + 2]), end=', ')
        print("1-NN real accuracy:" + str(score[i * 7 + 3]), end=', ')
        print("1-NN fake accuracy:" + str(score[i * 7 + 4]), end=', ')
        print("1-NN precision:" + str(score[i * 7 + 5]), end=', ')
        print("1-NN recall:" + str(score[i * 7 + 6]))
    print("Inception score:" + str(score[28]))
    print("Mode score:" + str(score[29]))
    print("FID:" + str(score[30]))
    print("Correlation scores:" + str(score[31]))
    print("Significant Real:" + str(score[32]))
    print("Significant Fake:" + str(score[33]))
    print("Real over 0_5:" + str(score[34]))
    print("Fake over 0_5:" + str(score[35]))


def get_score_names():
    '''传入的scores是个list，包含了每k轮训练的score结果'''
    names = []
    for i in range(4):
        if i == 0:
            feature = 'Raw'
        elif i == 1:
            feature = 'Conv'
        elif i == 2:
            feature = 'Fc'
        else:
            feature = 'Softmax'
        names.append(feature + ' Wasserstein distance')
        names.append(feature + ' MMD')
        names.append(feature + ' 1-NN accuracy')
        names.append(feature + ' 1-NN real accuracy')
        names.append(feature + ' 1-NN fake accuracy')
        names.append(feature + ' 1-NN precision')
        names.append(feature + ' 1-NN recall')
    names.append('Inception')
    names.append('Mode')
    names.append('FID')
    names.append('Correlation')
    names.append('Significant Real')
    names.append('Significant Fake')
    names.append('Real over 0_5')
    names.append('Fake over 0_5')

    return names


if __name__ == '__main__':

    real_train_path = '/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/train/'
    real_test_path = '/home/cbd109/Users/hmq/GANDatasets/LK_rest/B/test/'
    fake_path = '/home/cbd109/Users/hmq/codes/pix2pix/results/IF_GAN/test_latest/npys/'
    normalizer = DataNormalizer(None)
    dataset = EEG(real_train_path, isreal=True, transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
    #dataset = EEG(fake_path, isreal=False, isIF=True, normalizer=normalizer,
    #              transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    get_max_min_normalize_params(dataloader, 'IF_GAN_maxmin.npy')
    #score = compute_score_raw(1, real_test_path, fake_path, "IF_GAN_metrics.npy", normalizer=normalizer)
    #print_score(score)
