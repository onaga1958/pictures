from keras.layers import Convolution2D, Dense, Flatten, MaxPool2D
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from os.path import abspath, dirname, join
from math import ceil
from copy import copy

import os
import numpy as np


IMG_SIZE = (100, 100, 3)


class Datagen:
    def __init__(self, img_dir, train_gt=None, batch_size=32):
        self.index = 0
        self.file_names = os.listdir(img_dir)
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.y = train_gt
        if self.y is None:
            self.sizes = np.zeros((len(self.file_names), 2))

    def __iter__(self):
        return self

    def _full_name(self, name):
        return join(self.img_dir, name)

    def get_points_number(self):
        if self.y is None:
            raise('You can\'t get points number from test Datagen')
        else:
            return self.y[self.file_names[0]].shape[0]

    def _next_index(self, old_index):
        return (old_index + self.batch_size) % len(self.file_names)

    def __next__(self):
        old_index = copy(self.index)
        self.index = self._next_index(old_index)

        end_batch = old_index + self.batch_size
        file_names_batch = self.file_names[old_index:end_batch]
        if self.y is not None and end_batch > len(self.file_names):
            file_names_batch += self.file_names[:self._next_index(old_index)]

        if self.y is not None:
            y_batch = [self.y[file_name] for file_name in file_names_batch]

        images_batch = [imread(self._full_name(file_name))
                        for file_name in file_names_batch]

        if self.y is None:
            sizes = np.array([image.shape[:2] for image in images_batch])
            self.sizes[old_index:end_batch] = sizes

        images_batch = [resize(image, IMG_SIZE, mode='reflect')
                        for image in images_batch]
        if self.y is not None:
            return np.array(images_batch), np.array(y_batch)
        else:
            return np.array(images_batch)


def _init_model(units, layers_in_level, levels, denses, filters, kernel_size,
                dense_size, kernel_initializer, kernel_regularizer, dropout,
                activation, filters_multiplicator):
    model = Sequential()

    for i in range(levels):
        for j in range(layers_in_level):
            params = {'filters': filters * filters_multiplicator**i,
                      'kernel_size': kernel_size,
                      'padding': 'same',
                      'kernel_initializer': kernel_initializer,
                      'activation': activation,
                      'kernel_regularizer': kernel_regularizer,
                      }
            if i == 0 and j == 0:
                params['input_shape'] = IMG_SIZE

            model.add(Convolution2D(**params))
        model.add(MaxPool2D(padding='same'))
        model.add(BatchNormalization())

    model.add(Flatten())
    for i in range(denses):
        params = {'kernel_initializer': kernel_initializer,
                  'kernel_regularizer': kernel_regularizer, }
        if i == denses - 1:
            params['units'] = units
        else:
            params['activation'] = activation
            params['units'] = dense_size
        model.add(Dense(**params))
        if i != denses - 1:
            model.add(Dropout(dropout))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(decay=1e-3),
                  metrics=['mse'])
    return model


def _get_images_from_directory(img_dir):
    file_names = os.listdir(img_dir)
    images = [imread(join(img_dir, file_name))
              for file_name in file_names]
    sizes = map(np.shape, images)
    images = [resize(image, IMG_SIZE, mode='reflect')
              for image in images]
    return np.array(images), sizes, file_names


def _rescale_answers(answers, sizes, straight):
    shape = answers.shape
    reshaped_answers = np.zeros((shape[0], 2, shape[1] // 2))
    reshaped_answers[:, 0] = answers[:, 0::2]
    reshaped_answers[:, 1] = answers[:, 1::2]
    reshaped_answers = reshaped_answers.transpose([2, 0, 1])
    rescale_to = np.array(IMG_SIZE[:2])

    multiplyer = (sizes / rescale_to)
    if straight:
        reshaped_answers /= multiplyer
    else:
        reshaped_answers *= multiplyer

    answers[:, 0::2] = reshaped_answers[:, :, 0].T
    answers[:, 1::2] = reshaped_answers[:, :, 1].T

    return answers


def _get_data(train_gt, train_img_dir):
    images, sizes, file_names = _get_images_from_directory(train_img_dir)
    answers = np.array([train_gt[file_name] for file_name in file_names])
    return images, _rescale_answers(answers, sizes, True)


def train_detector(train_gt, train_img_dir, fast_train, validation=0.0):
    batch_size = 32
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'facepoints_model.hdf5')
    epochs = 1 if fast_train else 200
    epochs_batch = 10

    if not fast_train:
        X_train, y_train = _get_data(train_gt, train_img_dir)
        points_number = y_train.shape[1]
    else:
        datagen = Datagen(train_img_dir, train_gt, batch_size)
        points_number = datagen.get_points_number()

    if not fast_train and validation:
        split_reslut = train_test_split(X_train, y_train, test_size=validation,
                                        random_state=42)
        X_train, X_test, y_train, y_test = split_reslut

    if fast_train or not os.path.exists(model_path):
        model = _init_model(points_number, levels=3, layers_in_level=2,
                            filters=32, denses=2, dense_size=512,
                            filters_multiplicator=2,
                            kernel_size=3,
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=l2(1e-2), dropout=0.5,
                            activation='elu')
    else:
        model = load_model(model_path)

    for i in range(0, epochs, epochs_batch):
        try:
            if not fast_train:
                model.fit(X_train, y_train,
                          batch_size=batch_size,
                          validation_data=(X_test, y_test),
                          epochs=epochs_batch + i,
                          initial_epoch=i)
                save_model(model, model_path)
            else:
                model.fit_generator(datagen, steps_per_epoch=1,
                                    epochs=1)

        except MemoryError as e:
            print('memory error!')
            break

        if not fast_train and validation:
            loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
            print(loss_and_metrics)


def detect(model, test_img_dir):
    batch_size = 256

    datagen = Datagen(test_img_dir, batch_size=batch_size)
    steps = ceil(len(datagen.file_names) / batch_size)

    answers = model.predict_generator(datagen, steps=steps,
                                      max_queue_size=1)

    answers = _rescale_answers(answers, datagen.sizes, False)
    answers = {file_name: answer
               for file_name, answer in zip(datagen.file_names, answers)}
    return answers
