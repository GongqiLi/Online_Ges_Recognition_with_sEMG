# coding = utf8

from collections import deque
import numpy as np
import torch
from scipy import signal


def band_filter(data):
    [b, a] = signal.butter(3, 0.00001, btype='highpass')
    filtered_data = signal.filtfilt(b, a, data, padlen=0)
    [b, a] = signal.butter(3, 0.3, btype='lowpass')
    filtered_data = signal.filtfilt(b, a, filtered_data, padlen=0)
    return np.abs(filtered_data)


class OnlineGestureDetection(object):
    def __init__(self, model_path, param):
        self.model = torch.load(model_path)
        self.model.eval()
        self.buffer = deque(maxlen=param['buffer_length'])
        self.threshold = param['threshold']
        self.window_length = param['window_length']
        self.effective_zone = param['effective_zone']
        self.previous_prediction = None
        self.filter = param['filter']

    def take_in_data(self, data):
        self.buffer.append(data)

    def take_in_multiple_data(self, data):
        self.buffer.extend(data)

    def predict(self):
        softmax_scores = [0.125] * 8
        data = np.array(self.buffer)
        if filter:
            filtered_data = np.zeros([len(data), len(data[0])])
            for i in range(len(data[0])):
                filtered_data[:, i] = np.array(band_filter(np.array(data)[:, i]))
        else:
            filtered_data = np.subtract(data, 118)

        if len(filtered_data) > self.effective_zone:
            filtered_data = filtered_data[(-1 - self.effective_zone):-1]
            filtered_data = torch.from_numpy(np.expand_dims(np.expand_dims(filtered_data.T, axis=0), axis=0)).float()
            scores = self.model(filtered_data)
            _, predict = torch.max(scores.data, 1)
            softmax_scores = np.array((torch.softmax(scores.data.detach(), 1)[0]))
            predict = predict.numpy()[0]
            self.previous_prediction = predict
        else:
            predict = self.previous_prediction
        return predict, softmax_scores


