import numpy as np
import scipy.ndimage as ndi


def mse(img1, img2):
    return np.mean((img1 - img2)**2)


def cross_cor(img1, img2):
    return -np.sum(img1 * img2) / np.sqrt(np.sum(img1**2)*np.sum(img2**2))


def shift(img, i, j):
    img = img[i:] if i >= 0 else img[:i]
    img = img[:, j:] if j >= 0 else img[:, :j]
    return img


def preprocessing(img):
    remainder = img.shape[0] % 3
    if remainder != 0:
        img = img[:-remainder]

    img = img.reshape(3, img.shape[0] // 3, img.shape[1])
    pad = 0.05
    x_pad = int(pad * img.shape[1])
    y_pad = int(pad * img.shape[2])
    img = img[:, x_pad:-x_pad, y_pad:-y_pad]
    return img


def resolution_reduction(img):
    x = np.arange(0, img.shape[1], 2)
    y = np.arange(0, img.shape[2], 2)
    img = img[:, x]
    return img[:, :, y]


def find_optimal_shift(img, unknown_channels, known_channel, metric):
    pyramid = [img]

    while pyramid[-1].shape[1] > 500:
        pyramid.append(resolution_reduction(pyramid[-1]))
    start_shift = np.zeros((2, 2), dtype=np.int)

    max_shift = 15
    for i, img_version in enumerate(reversed(pyramid)):
        start_shift *= 2
        start_shift = searching_shift(img_version, unknown_channels,
                                      known_channel, metric, max_shift,
                                      start_shift)
        max_shift = 1

    return start_shift


def searching_shift(img, unknown_channels, known_channel, metric,
                    max_shift, start_shift):
    shifts = []

    for ss, channel in zip(start_shift, unknown_channels):
        min_metric = 10 ** 10
        curr_shift = [0, 0]

        for i in range(-max_shift + ss[0], max_shift + 1 + ss[0]):
            for j in range(-max_shift + ss[1], max_shift + 1 + ss[1]):
                metr = metric(shift(img[channel], i, j),
                              shift(img[known_channel], -i, -j))
                if metr < min_metric:
                    min_metric = metr
                    curr_shift = [i, j]

        shifts.append(curr_shift)

    return np.array(shifts)


def align(img, g_coord, metric=cross_cor):
    img = preprocessing(img)
    unknown_channels = [0, 2]
    known_channel = 1

    shifts = find_optimal_shift(img, unknown_channels, known_channel, metric)

    for sh, channel in zip(shifts, unknown_channels):
        img[channel] = ndi.shift(img[channel], sh)

    original_shape = int(img.shape[1] * 10 / 9)
    shifts[0][0] -= original_shape
    shifts[1][0] += original_shape

    return [img.transpose([1, 2, 0])] + [[gc + sh
                                          for gc, sh in zip(g_coord, shs)]
                                         for shs in shifts]
