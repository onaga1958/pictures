import numpy as np
import scipy.ndimage as ndi


def mse(img1, img2):
    return np.mean((img1 - img2)**2)


def cross_cor(img1, img2):
    return -np.sum(img1 * img2) / np.sqrt(np.sum(img1**2)*np.sum(img2**2))


def cutting(img, shifts, unknown_channels):
    new_img = [i for i in img]

    for dim in range(2):
        for sh, channel in enumerate(unknown_channels):
            another_channel = 2 if channel == 0 else 0
            first = shifts[sh][dim]*(1 - dim)
            second = shifts[sh][dim] * dim

            if shifts[0][dim] * shifts[1][dim] <= 0:
                new_img[another_channel] = one_shift(new_img[another_channel],
                                                     -first, -second)
                new_img[channel], new_img[1] = shift(new_img, first,
                                                     second, [channel, 1])
            else:
                if abs(shifts[sh][dim]) > abs(shifts[(sh + 1)%2][dim]):
                    new_img[channel], new_img[1] = shift(new_img, first,
                                                         second, [channel, 1])
                else:
                    new_img[channel] = one_shift(new_img[channel], first, second)

                    rest_shift =  shifts[sh][dim] - shifts[(sh + 1)%2][dim]
                    new_img[channel] = one_shift(new_img[channel],
                                                 rest_shift*(1 - dim),
                                                 rest_shift*dim)
    return np.array(new_img).transpose([1, 2, 0])


def align(img, g_coord, metric=cross_cor):
    remainder = img.shape[0] % 3
    if remainder != 0:
        img = img[:-remainder]

    img = img.reshape(3, img.shape[0] // 3, img.shape[1])
    pad = 0.05
    x_pad = int(0.05 * img.shape[1])
    y_pad = int(0.05 * img.shape[2])
    img = img[:, x_pad:-x_pad, y_pad:-y_pad]
    shifted_img = np.array([i for i in img])

    max_shift = 15
    shifts = []
    unknown_channels = [0, 2]
    metrics = np.zeros((31, 31))

    for channel in unknown_channels:
        for i in range(-max_shift, max_shift + 1):
            for j in range(-max_shift, max_shift + 1):
                shifted_img[channel] = ndi.shift(img[channel], (i, j))
                metr = metric(shifted_img[channel], img[1])
                metrics[i+max_shift][j+max_shift] = metr

        optimal = np.where(metrics == metrics.min())
        print(optimal)
        shifts.append([optimal[0][0] - max_shift, optimal[1][0] - max_shift])

    align_img = cutting(img, shifts, unknown_channels)

    original_shape = int(img.shape[1] * 10 / 9)
    shifts[0][0] -= original_shape
    shifts[1][0] += original_shape

    return [align_img] + [[gc + sh for gc, sh in zip(g_coord, shs)]
                          for shs in shifts]
