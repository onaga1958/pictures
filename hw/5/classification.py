import numpy as np
import keras.layers as L
import os

from copy import copy
from keras.applications.resnet50 import ResNet50
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from math import ceil
from os.path import abspath, dirname, join
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


IMG_SIZE = (224, 224, 3)
N_CLASSES = 50


class Datagen:
    def __init__(self, img_dir, train_gt=None, batch_size=32):
        self.index = 0
        self.file_names = os.listdir(img_dir)
        self.batch_size = batch_size
        self.img_dir = img_dir
        if train_gt is not None:
            self.y = _get_answers(train_gt, self.file_names)
        else:
            self.y = None

    def __iter__(self):
        return self

    def _full_name(self, name):
        return join(self.img_dir, name)

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
            y_batch = self.y[old_index:end_batch]

        images_batch = [imread(self._full_name(file_name))
                        for file_name in file_names_batch]

        images_batch = [resize(image, IMG_SIZE, mode='reflect')
                        for image in images_batch]
        if self.y is not None:
            return np.array(images_batch), np.array(y_batch)
        else:
            return np.array(images_batch)


def _get_images_from_directory(img_dir):
    file_names = os.listdir(img_dir)
    images = [imread(join(img_dir, file_name))
              for file_name in file_names]
    images = [resize(image, IMG_SIZE, mode='reflect')
              for image in images]
    return np.array(images), file_names


def _get_answers(train_gt, file_names):
    answers = np.zeros((len(file_names), N_CLASSES))
    answers_ones_indexes = np.array([train_gt[file_name]
                                     for file_name in file_names])
    answers[np.arange(len(file_names)), answers_ones_indexes] = 1
    return answers


def _get_data(train_gt, train_img_dir):
    images, file_names = _get_images_from_directory(train_img_dir)
    answers = _get_answers(train_gt, file_names)
    return images, answers


def _init_model():
    basemodel = ResNet50(include_top=False, input_shape=IMG_SIZE,
                         pooling='avg')
    base_out = basemodel.output

    predictions = L.Dense(N_CLASSES, activation='softmax',
                          kernel_regularizer=l2(1e-3))(base_out)
    model = Model(inputs=basemodel.input, output=predictions)

    for layer in basemodel.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=1e-3, decay=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_classifier(train_gt, train_img_dir, fast_train, validation=0.2):
    batch_size = 32
    epochs = 1 if fast_train else 50
    epochs_batch = 1 if fast_train else 5
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'birds_model.hdf5')

    if not fast_train:
        X_train, y_train = _get_data(train_gt, train_img_dir)
        datagen = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True)
        datagen = datagen.flow(X_train, y_train, batch_size=batch_size)
        steps_per_epoch = ceil(len(X_train) / batch_size)
    else:
        datagen = Datagen(train_img_dir, train_gt, batch_size)
        steps_per_epoch = 3

    if not fast_train and validation:
        split_reslut = train_test_split(X_train, y_train, test_size=validation,
                                        random_state=42)
        X_train, X_test, y_train, y_test = split_reslut
        validation_data = (X_test, y_test)
    else:
        validation_data = None

    if fast_train or not os.path.exists(model_path):
        model = _init_model()
    else:
        model = load_model(model_path)

    for i in range(0, epochs, epochs_batch):
        model.fit_generator(datagen,
                            max_queue_size=1,
                            epochs=epochs_batch + i,
                            initial_epoch=i,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_data)
        if not fast_train:
            save_model(model, model_path)


def classify(model, test_img_dir):
    batch_size = 32

    datagen = Datagen(test_img_dir, batch_size=batch_size)
    steps = ceil(len(datagen.file_names) / batch_size)

    answers = model.predict_generator(datagen, steps=steps,
                                      max_queue_size=1)

    answers = {file_name: np.argmax(answer)
               for file_name, answer in zip(datagen.file_names, answers)}
    return answers
