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
    x_pad = int(0.05 * img.shape[1])
    y_pad = int(0.05 * img.shape[2])
    img = img[:, x_pad:-x_pad, y_pad:-y_pad]
    return img


def find_optimal_shift(img, unknown_channels, known_channel, metric):
    max_shift = 15
    shifts = []
    metrics = np.zeros((31, 31))

    for channel in unknown_channels:
        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):
                metr = metric(shift(img[channel], i, j),
                              shift(img[known_channel], -i, -j))
                metrics[i+max_shift][j+max_shift] = metr

        optimal = np.where(metrics == metrics.min())
        print(optimal)
        shifts.append([optimal[0][0] - max_shift,
                       optimal[1][0] - max_shift])

    return shifts


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

    return [img.transpose([1, 2, 0])] + [[gc + sh for gc, sh in zip(g_coord, shs)]
                     for shs in shifts]
