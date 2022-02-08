from util.eeg_tools import load_from_ndarray, IF_to_eeg
import cv2
import numpy as np


class EEGBlock:

    def __init__(self, paths, normalizer=None):
        # 保存片段的宽度（信道数）和长度（采样点数
        data = []
        for path in paths:
            data.append(load_from_ndarray(path))
        self.data = np.vstack(data)

    def size(self):
        return self.data.shape

    def crop(self, left, upper, right, lower):
        print("Before cropping: " + str(self.data))
        print("Origin shape: " + str(self.size()))
        self.data = self.data[upper:lower, left:right]
        print("Cropped: " + str(self.data))
        return self

    def resize(self, w, h, method=cv2.INTER_CUBIC): #默认插值：INTER_CUBIC
        new_size = (h, w)
        self.data = cv2.resize(self.data, dsize=new_size, interpolation=method)
        return self

    def transpose(self, method):
        self.data = np.flip(self.data, 1)
        return self

    def get_data(self):
        return self.data

    def set_data(self, v):
        self.data = v
