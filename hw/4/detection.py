from keras.layers import Convolution2D, Dense, Activation, Flatten, MaxPool2D
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from keras.models import Sequential, save_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from os.path import abspath, dirname, join
from matplotlib.pyplot import show, imshow

import os
import numpy as np


IMG_SIZE = (100, 100, 3)


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
                  optimizer=Adam())
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


def train_detector(train_gt, train_img_dir, fast_train, validation=0.1):
    X_train, y_train = _get_data(train_gt, train_img_dir)
    batch_size = 32
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'facepoints_model.hdf5')
    epochs = 1 if fast_train else 10
    model = _init_model(y_train.shape[1], levels=3, layers_in_level=2,
                        filters=64, denses=3, dense_size=1024, kernel_size=3,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(5e-6),
                        activation='elu')

    if validation:
        split_reslut = train_test_split(X_train, y_train, test_size=validation)
        X_train, X_test, y_train, y_test = split_reslut
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=0,
                                 width_shift_range=0,
                                 height_shift_range=0,
                                 horizontal_flip=False,)
    datagen.fit(X_train)
    for _ in range(epochs):
        model.fit_generator(datagen.flow(X_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=1,
                            workers=4)
        if not fast_train:
            save_model(model, model_path)

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
