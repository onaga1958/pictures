from keras.layers import Convolution2D, Dense, Activation, Flatten, MaxPool2D
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from os.path import abspath, dirname, join
from matplotlib.pyplot import show, imshow
from copy import copy

import os
import threading
import numpy as np
import scipy.ndimage as ndi


IMG_SIZE = (100, 100, 3)


# ImageDataGenerator code was partially taken from here
# https://www.kaggle.com/hexietufts/easy-to-use-keras-imagedatagenerator/code
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class ImageDataGenerator:
    def __init__(self, featurewise_center=False,
                 featurewise_std_normalization=False,
                 row_shift_range=0, col_shift_range=0,
                 horizontal_flip=False, vertical_flip=False):
        self.__dict__.update(locals())
        self.channel_index = 3
        self.row_index = 1
        self.col_index = 2
        self.fill_mode = 'nearest'

    def flow(self, X, y, batch_size=32):
        return Iterator(X, y, self, batch_size)

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        # Maybe should swap 0::2 and 1::2
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        new_y = copy(y)

        if self.row_shift_range:
            tx = np.random.randint(-self.row_shift_range,
                                   self.row_shift_range + 1)
        else:
            tx = 0

        if self.col_shift_range:
            ty = np.random.randint(-self.col_shift_range,
                                   self.col_shift_range + 1)
        else:
            ty = 0

        padded = np.pad(x, ((self.row_shift_range, self.row_shift_range),
                            (self.col_shift_range, self.col_shift_range),
                            (0, 0)),
                        mode='edge')
        x = padded[-tx + self.row_shift_range: -tx + x.shape[0]
                   + self.row_shift_range,
                   -ty + self.col_shift_range: -ty + x.shape[1]
                   + self.col_shift_range]
        new_y[1::2] += tx
        new_y[0::2] += ty

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                new_y[0::2] = x.shape[img_col_index] - new_y[0::2] - 1

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                new_y[1::2] = x.shape[img_row_index] - new_y[1::2] - 1

        return x, new_y

    def fit(self, X):
        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)

    def standartize(self, x):
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= self.std + 1e-10
        return x


class Iterator:
    def __init__(self, X, y, image_data_generator, batch_size):
        self.__dict__.update(locals())
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(X.shape[0], batch_size)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size):
        self.reset()
        while True:
            if self.batch_index == 0:
                index_array = np.random.permutation(N)
            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0

            batch_end = current_index + current_batch_size
            yield index_array[current_index:batch_end], current_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            index_array, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.X.shape[1:])
        batch_y = np.zeros((current_batch_size, self.y.shape[1]))
        for batch_ind, global_ind in enumerate(index_array):
            x = self.X[global_ind]
            x = self.image_data_generator.standartize(x)
            y = self.y[global_ind]
            x, y = self.image_data_generator.random_transform(x, y)
            batch_x[batch_ind] = x
            batch_y[batch_ind] = y
        return batch_x, batch_y


def _init_model(units, layers_in_level, levels, denses, filters, kernel_size,
                dense_size, kernel_initializer, kernel_regularizer,
                activation):
    model = Sequential()

    for i in range(levels):
        for j in range(layers_in_level):
            params = {'filters': filters * 2**i,
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
                  'kernel_regularizer': kernel_regularizer,
                  'activation': activation}
        if i == denses - 1:
            params['units'] = units
        else:
            params['units'] = dense_size
        model.add(Dense(**params))
        if i != denses - 1:
            model.add(Dropout(0.25))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(),
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
    for j, size in enumerate(sizes):
        for i in range(2):
            multiplyer = size[i] / IMG_SIZE[i]
            if straight:
                multiplyer = 1 / multiplyer
            answers[j, i::2] *= multiplyer
    return answers


def _get_data(train_gt, train_img_dir):
    images, sizes, file_names = _get_images_from_directory(train_img_dir)
    answers = np.array([train_gt[file_name] for file_name in file_names])
    return images, _rescale_answers(answers, sizes, True)


def train_detector(train_gt, train_img_dir, fast_train, validation=0.0):
    X_train, y_train = _get_data(train_gt, train_img_dir)
    batch_size = 32
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'facepoints_model.hdf5')
    epochs = 1 if fast_train else 40
    if fast_train or not os.path.exists(model_path):
        model = _init_model(y_train.shape[1], levels=3, layers_in_level=2,
                            filters=64, denses=3, dense_size=512, kernel_size=3,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4),
                            activation='elu')
    else:
        model = load_model(model_path)

    if validation:
        split_reslut = train_test_split(X_train, y_train, test_size=validation)
        X_train, X_test, y_train, y_test = split_reslut
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 row_shift_range=10,
                                 col_shift_range=10,
                                 horizontal_flip=True,)
    datagen.fit(X_train)
    X_test = datagen.standartize(X_test)

    for i in range(epochs):
        if not fast_train:
            print(str(i + 1) + ' epoch')
        try:
            model.fit_generator(datagen.flow(X_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                epochs=1,
                                workers=4)
            if not fast_train:
                save_model(model, model_path)
        except MemoryError as e:
            print('memory error!')
            break

        if validation:
            loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
            print(loss_and_metrics)


def detect(model, test_img_dir):
    X_test, sizes, file_names = _get_images_from_directory(test_img_dir)
    answers = model.predict(X_test, batch_size=128)
    answers = _rescale_answers(answers, sizes, False)
    answers = {file_name: answer
               for file_name, answer in zip(file_names, answers)}
    return answers
