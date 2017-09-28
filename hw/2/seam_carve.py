import numpy as np


def get_brightness(point):
    return 0.299*point[0] + 0.587*point[1] + 0.114*point[2]


def get_derivative(img, x, y):
    if x != 0 and x != img.shape[0] - 1:
        return img[x + 1][y] - img[x - 1][y]
    else:
        if x == 0:
            return img[x + 1][y] - img[x][y]
        else:
            return img[x][y] - img[x - 1][y]


def get_grad(img, x, y):
    x_der = get_derivative(img, x, y)
    y_der = get_derivative(img.transpose([1, 0, 2]), y, x)
    return np.sqrt(x_der ** 2 + y_der ** 2)


def count_energy(img):
    return np.array([[get_grad(img, x, y)
                      for x in range(img.shape[0])]
                     for y in range(img.shape[1])])


def count_energy_among_seam(energy):
    energy_among_seam = np.zeros_like(energy)
    energy_among_seam[0] = energy[0]

    for x in range(1, energy.shape[0]):
        for y in range(energy.shape[1]):
            possible_y = range(max(y - 1, 0), min(y + 2, energy.shape[1]))
            min_energy = min([energy_among_seam[x - 1][curr_y]
                              for curr_y in possible_y]])
            energy_among_seam[x][y] = min_energy + energy[x][y]

    return energy_among_seam


def find_seam_with_least_energy(energy):
    previous_y = np.argmin(energy[-1], axis=0)
    seam_mask = np.zeros_like(energy)
    seam_mask[-1][previous_y] = 1

    for x in range(energy.shape[0] - 2, -1, -1):

        y_max = energy.shape[1] - 1
        y_to_analyze = [np.arange(max(y - 1, 0), min(y + 1, y_max) + 1)
                        for y in previous_y]
        analyzed_area = [energy[x][y, color]
                         for color, y in enumerate(y_to_analyze)]
        print(analyzed_area)
        previous_y += (np.argmin(analyzed_area, axis=1) +
                       np.array([-1 if y != 0 else 0 for y in previous_y]))
        seam_mask[x][[previous_y, np.arange(3)]] = 1

    return seam_mask


def remove_seam_with_least_energy(img, energy, mask):
    seam_mask = find_seam_with_least_energy(energy)
    return img[1 - seam_mask], mask[1 - seam_mask], seam_mask


def seam_carve(img, command, mask=None):
    energy = count_energy(img)
    if command == 'horizontal shrink':
        result = remove_seam_with_least_energy(img, energy, mask)
    if command == 'vertical shrink':
        result = remove_seam_with_least_energy(img.transpose([1, 0, 2]), energy.transpose([1,0,2]), mask.transpose([1,0,2]))
        result = [res.transpose([1, 0, 2]) for res in result]
    return result
