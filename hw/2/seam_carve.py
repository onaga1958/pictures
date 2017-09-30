import numpy as np


def get_derivative(matrix):
    return np.concatenate((matrix[1:2] - matrix[:1],
                           matrix[2:] - matrix[:-2],
                           matrix[-1:] - matrix[-2:-1]))


def count_energy(img, mask):
    max_energy = img.shape[0] * img.shape[1] * 256
    brightness_matrix = (0.299*img[:, :, 0] +
                         0.587*img[:, :, 1] +
                         0.114*img[:, :, 2])

    x_derivative = get_derivative(brightness_matrix)
    y_derivative = get_derivative(brightness_matrix.transpose()).transpose()
    energy = np.sqrt(x_derivative**2 + y_derivative**2)
    if mask is not None:
        energy[mask == 1] += max_energy
        energy[mask == -1] -= max_energy
    return energy


def count_energy_among_seam(energy):
    energy_among_seam = np.zeros_like(energy)
    energy_among_seam[0] = energy[0]
    for x in range(1, energy.shape[0]):
        min_energy = energy_among_seam[x - 1, :2].min()
        energy_among_seam[x, 0] = min_energy + energy[x, 0]
        for y in range(1, energy.shape[1] - 1):
            min_energy = energy_among_seam[x - 1, y - 1:y + 2].min()
            energy_among_seam[x, y] = min_energy + energy[x, y]
        min_energy = energy_among_seam[x - 1, -2:].min()
        energy_among_seam[x, -1] = min_energy + energy[x, -1]
    return energy_among_seam


def find_seam_with_least_energy(energy):
    seam_mask = np.zeros_like(energy, dtype=np.int16)
    previous_y = np.argmin(energy[-1])
    seam_mask[-1, previous_y] = 1

    for x in range(energy.shape[0] - 2, -1, -1):
        y_max = energy.shape[1] - 1
        if previous_y != 0:
            max_y = min(previous_y + 2, energy.shape[1])
            step_directions = np.argmin(energy[x, previous_y - 1:max_y]) - 1
        else:
            step_directions = np.argmin(energy[x, :2])

        previous_y += step_directions
        seam_mask[x, previous_y] = 1

    return seam_mask


def changed_shape(obj, diff, axis):
    shape = list(obj.shape)
    shape[1 - axis] += diff
    if axis == 1:
        shape[0], shape[1] = shape[1], shape[0]
    return shape


def to_img(mask):
    return mask.reshape(mask.shape + (1,))


def to_mask(imged_mask):
    return imged_mask[:, :, 0]


def remove_seam_from_img(img, seam_mask, axis):
    shape = changed_shape(img, -1, axis)
    img = img.transpose([2, axis, 1 - axis])
    if axis == 1:
        seam_mask = seam_mask.transpose()
    img = img[:, seam_mask == 0].transpose().reshape(shape)
    if axis == 1:
        img = img.transpose([1, 0, 2])
    return img


def add_seam_to_img(img, seam_mask, axis):
    shape = changed_shape(img, 1, axis)
    transpose_shape = [shape[2], shape[0], shape[1]]
    img = img.transpose([2, axis, 1 - axis])
    if axis == 1:
        seam_mask = seam_mask.transpose()
    zeros = np.zeros([shape[2], seam_mask.shape[0], 1], dtype=np.int16)

    cum_mask = np.cumsum(seam_mask, axis=1)

    new_img = np.concatenate((img * (1 - cum_mask), zeros), axis=2)
    new_img += np.concatenate((zeros, img * cum_mask), axis=2)
    new_img += np.concatenate((img * seam_mask, zeros), axis=2)

    return new_img.transpose([1 + axis, 2 - axis, 0])


def seam_carve(img, command, mask=None):
    old_mask = mask
    command = command.split(' ')
    axis = 0 if command[0] == 'horizontal' else 1

    energy = count_energy(img, mask)
    if axis:
        energy = energy.transpose()

    energy = count_energy_among_seam(energy)
    seam = find_seam_with_least_energy(energy)
    if axis:
        seam = seam.transpose()

    if command[1] == 'shrink':
        img = remove_seam_from_img(img, seam, axis)
        if mask is not None:
            mask = to_mask(remove_seam_from_img(to_img(mask), seam, axis))
    else:
        img = add_seam_to_img(img, seam, axis)
        if mask is not None:
            mask = to_mask(add_seam_to_img(to_img(mask), seam, axis))

    return img, mask, seam
