import numpy as np
import os
import csv
from functools import partial, reduce
from keras.utils import Sequence
from sklearn.utils import shuffle
import cv2


add_correction = lambda c, x: x + c
flip_image = lambda image: np.fliplr(image)
flip_steering = lambda steering: -1 * steering


def build_fit_dataset(data_path):
    dataset = []
    correction = .2
    data_path = os.path.abspath(data_path)

    with open(os.path.join(data_path, "driving_log.csv")) as read_file:
        reader = csv.reader(read_file)
        next(reader, None)
        for center,left,right,steering,throttle,brake,speed in reader:
            dataset.append((os.path.join(data_path, center.strip()), [cv2.imread], float(steering), []))
            dataset.append((os.path.join(data_path, left.strip()), [cv2.imread], float(steering), [partial(add_correction, correction)]))
            dataset.append((os.path.join(data_path, right.strip()), [cv2.imread], float(steering), [partial(add_correction, -1 * correction)]))
            dataset.append((os.path.join(data_path, center.strip()), [cv2.imread, flip_image], float(steering), [flip_steering]))
            dataset.append((os.path.join(data_path, left.strip()), [cv2.imread, flip_image], float(steering), [partial(add_correction, correction), flip_steering]))
            dataset.append((os.path.join(data_path, right.strip()), [cv2.imread, flip_image], float(steering), [partial(add_correction, -1 * correction), flip_steering]))
    return dataset


class CarNDBehavioralCloningSequence(Sequence):
        def __init__(self, dataset, batch_size):
            self.dataset = shuffle(dataset)
            self.batch_size = batch_size
        def __len__(self):
            return len(self.dataset) // self.batch_size
        def __getitem__(self, idx):
            batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
            X = np.array([reduce(lambda x, func: func(x), sample[1], sample[0]) for sample in batch])
            y = np.array([reduce(lambda x, func: func(x), sample[3], sample[2]) for sample in batch])
            return X, y
        def on_epoch_end(self):
            pass


def dataset_generator(dataset, batch_size):
    while 1:
        shuffle_dataset = shuffle(dataset)
        for start in range(0, len(shuffle_dataset), batch_size):
            batch = shuffle_dataset[start: start + batch_size]
            X = np.array([reduce(lambda x, func: func(x), sample[1], sample[0]) for sample in batch])
            y = np.array([reduce(lambda x, func: func(x), sample[3], sample[2]) for sample in batch])
            yield X, y
