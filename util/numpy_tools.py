import numpy as np
import matplotlib.pyplot as plt
import os
import ntpath
import seaborn as sns


EXTENSIONS = [".npy"]

def is_npy_file(filename): #判断是否是上述图片格式
    return any(filename.endswith(extension) for extension in EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")): #把目录内的所有图片文件的路径+名称放进一个列表内
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npy_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def save_as_ndarray(filename, data_array): #直接把数据存为ndarray
    np.save(filename, data_array)
    # print("Successfully save file %s!" % filename)


def load_from_ndarray(filename): #从ndarray中读
    if not filename.endswith(".npy"):
        data = np.load(filename + ".npy")
    else:
        data = np.load(filename)
    return data


'''def draw_comparable_figure(s_start=0, s_end=None, ch_intv=0.002, **data_dict):
    plotting_list = [] #存储每个子图的y值
    title_list = []
    x = []
    for title, data in data_dict.items():
        if s_end is None or s_end > len(data[0]):
            s_end = len(data[0])
        tmp = data[:, s_start:s_end].T
        if ch_intv != 0:
            offset = np.arange(0, len(data) * ch_intv, ch_intv)
            tmp += offset
        plotting_list.append(tmp)
        title_list.append(title)
        x.append(np.array([i for i in range(s_end - s_start)]))
    fig = plt.figure()
    for i, to_plot in enumerate(plotting_list):
        ax = fig.add_subplot(len(plotting_list), 1, i + 1)
        ax.set_title(title_list[i])
        ax.plot(x[i], to_plot, linewidth=0.6)
    plt.show()'''
def draw_comparable_figure(xlabel, ylabel, sf, s_start=0, s_end=None, ch_intv=0.002, show=False, save_path=None, **data_dict):

    plotting_list = [] #存储每个子图的y值
    title_list = []
    x = []

    for title, data in data_dict.items():
        if s_end is None or s_end > len(data[0]):
            s_end = len(data[0])
        tmp = data[:, s_start:s_end].T
        if ch_intv != 0:
            offset = np.arange(0, len(data) * ch_intv, ch_intv)
            tmp += offset
        plotting_list.append(tmp)
        title_list.append(title)
        x.append(np.repeat(np.array([i for i in range(s_end - s_start)])[np.newaxis, :], tmp.shape[1], axis=0).T / sf)
    fig = plt.figure()

    for i, to_plot in enumerate(plotting_list):
        ax = fig.add_subplot(len(plotting_list), 1, i + 1)
        ax.set_title(title_list[i])
        # ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x[i], to_plot, linewidth=0.6)
    #print(plotting_list)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    elif show:
        plt.show()

    return fig


def plot_spectrogram(times, freqs, xlabel='Time step', ylabel='Frequency(Hz)',  vmin=None, vmax=None, show=True, save_path=None, **data_dict):
    """

    :param times: array-like time sequence, shape (n_times, )
    :param freqs: array_like of float, shape (n_freqs,)  list of output frequencies
    :param power: power to show
    :param vmin:
    :param vmax:
    :return: None
    """
    fig = plt.figure()
    # fig, ax = plt.subplots()
    for i, (subtitle, power) in enumerate(data_dict.items()):
        ax = fig.add_subplot(1, len(data_dict), i + 1)
        mesh = ax.pcolormesh(times, freqs, power,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.set(ylim=freqs[[0, -1]], xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(mesh, ax=ax)
    suptitle = os.path.basename(save_path).split('.')[0] if save_path is not None else ' '
    plt.suptitle(suptitle)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    elif show:
        plt.show()

    plt.close()


# def save_resized_npy(visuals, image_dir, image_path, save_size):
#     """
#     将G的输出resize为输入尺寸
#     :param visuals:
#     :param image_dir:
#     :param image_path:
#     :param save_size:
#     :return:
#     """
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)
#
#     for label, im_data in visuals.items():
#         npy_name = '%s_%s' % (name, label) # im_data是tensor
#         npy_data = im_data[0][0]
#         npy_data = npy_data.cpu().numpy()
#         new_size = (save_size[1], save_size[0])
#         resized_data = cv2.resize(npy_data, dsize=new_size, interpolation=cv2.INTER_CUBIC)
#         save_path = os.path.join(image_dir, npy_name)
#         np.save(save_path, resized_data)


def save_origin_npy(visuals, image_dir, image_path):
    """
    保存G的输出
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
        npy_name = '%s_%s' % (name, label) # im_data是tensor
        npy_data = im_data[0]
        npy_data = npy_data.cpu().numpy()
        save_path = os.path.join(image_dir, npy_name)
        np.save(save_path, npy_data)

def plot_mean_std(xticks, mean, std, title=None, xlabel=None, ylabel=None, show=False, save_dir=None):

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.errorbar(np.arange(len(mean)), mean, yerr=std, fmt="o")
    plt.xticks(np.arange(len(mean)), xticks)

    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + '.png'))


def plot_hist_distribution(data, bins, xlabel, ylabel, title, show=False, save_dir=None):
    '''
    把一维数据的分布通过直方图画出来
    :param data:
    :param bin:int or array like
    :return:
    '''
    sns.distplot(data, bins=bins, kde=True, norm_hist=True)
    sns.utils.axlabel(xlabel, ylabel)
    plt.title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + '.png'))
    if show:
        plt.show()
    plt.clf()
