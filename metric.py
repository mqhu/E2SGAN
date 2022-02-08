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

from util.eeg_tools import EEGSegment
import util.numpy_tools as nutil
import data.base_dataset as bd
from gansynth.normalizer import DataNormalizer


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
        segment.set_data(np.expand_dims(segment.get_data(), axis=0))
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

    #real max = 0.00226145, real min = -0.002839332
    data = img.get_data()
    max_v = 0.0014225434
    min_v = -0.0019580023
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
    ''' 我觉得可以不要normalization操作
    if model.find('vgg') >= 0 or model.find('resnet') >= 0:
        transform.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif model.find('inception') >= 0:
        transform[1] = transforms.Lambda(lambda img: bd.__square_size(img, 299))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        '''
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
                        input).mean(3).mean(2)  # 如果channel维是1的则不要squeeze，否则要squeeze
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

'''
def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1)
    print("1")
    X = X.cpu().numpy()
    print("2")
    print(X.shape)
    X2 = (X*X).sum(axis=1).reshape(nX, 1)
    Y = Y.view(nY, -1)
    Y = Y.cpu().numpy()
    print(Y.shape)
    print(Y.transpose(1, 0).shape)
    Y2 = (Y*Y).sum(axis=1).reshape(nY, 1)

    print("3")
    M = np.tile(X2, (1, nY)) + np.tile(Y2, (1, nX)).transpose((1, 0)) - \
        2 * np.matmul(X, Y.transpose(1, 0))
    print("4")
    del X, X2, Y, Y2

    if sqrt:
        M = ((M + np.fabs(M)) / 2).sqrt()

    return torch.from_numpy(M)'''


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
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
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
    # saveFolder_r和saveFolder_f分别是真图和假图的目录

    #sampleTrue(dataset, imageSize, dataroot, sampleSize, batchSize,
    #           saveFolder_r, workers=workers)
    #sampleFake(netG, nz, sampleSize, batchSize, saveFolder_f, )

    convnet_feature_saver = ConvNetFeatureSaver(model=conv_model,
                                                batchSize=batchSize, workers=workers)
    feature_r = convnet_feature_saver.save(saveFolder_r, isreal=True, isIF=False, normalizer=None)
    feature_f = convnet_feature_saver.save(saveFolder_f, isreal=False, isIF=True, normalizer=normalizer)

    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 7 + 3)
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
    #score[30] = fid(feature_r[3], feature_f[3]) # 有人说应该在conv层计算，即feature_r/f[1]
    score[30] = fid(feature_r[1], feature_f[1])
    np.save(save_name, score)
    return score


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


if __name__ == '__main__':

    real_test_path = "/home/cbd109/Users/hmq/GANDatasets/LK_one_channel/B/test/"
    real_train_path = "/home/cbd109/Users/hmq/GANDatasets/LK_one_channel/B/train/"
    fake_path = '/home/cbd109/Users/hmq/codes/pix2pix/results/IF_one_channel/test_latest/npys/'
    normalizer = DataNormalizer(None, '../one_channel.npy')
    #dataset = EEG(real_train_path, isreal=True, transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
    #dataset = EEG(fake_path, isreal=False, isIF=True, normalizer=normalizer,
    #              transforms=transforms.Lambda(lambda img: bd.__eegsegment_to_tensor(img)))
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    #get_max_min_normalize_params(dataloader, 'IF_one_channel_maxmin.npy')
    #calculate_means_and_var(dataloader, 'full_test_means_var.npy')
    score = compute_score_raw(1, real_test_path, fake_path, "IF_GAN_metrics.npy", normalizer=normalizer)
    print_score(score)
