import numpy as np


def get_derivative_x(matrix):
    return np.concatenate((matrix[1:2] - matrix[:1],
                           matrix[2:] - matrix[:-2],
                           matrix[-1:] - matrix[-2:-1]))


def get_derivative_y(matrix):
    return np.concatenate((matrix[:, 1:2] - matrix[:, :1],
                           matrix[:, 2:] - matrix[:, :-2],
                           matrix[:, -1:] - matrix[:, -2:-1]), axis=1)


def count_energy(img, mask):
    max_energy = img.shape[0] * img.shape[1] * 256
    brightness_matrix = (0.299*img[:, :, 0] +
                         0.587*img[:, :, 1] +
                         0.114*img[:, :, 2])

    x_derivative = get_derivative_x(brightness_matrix)
    y_derivative = get_derivative_y(brightness_matrix)
    energy = np.sqrt(x_derivative**2 + y_derivative**2)
    if mask is not None:
        energy[mask == 1] += max_energy
        energy[mask == -1] -= max_energy
    return energy


def get_possible_cord(cord, shape):
    return np.arange(max(cord - 1, 0), min(cord + 2, shape))


def count_energy_among_seam(energy):
    energy_among_seam = np.zeros_like(energy)
    energy_among_seam[0] = energy[0]
    for x in range(1, energy.shape[0]):
        for y in range(energy.shape[1]):
            possible_y = get_possible_cord(y, energy.shape[1])
            min_energy = energy_among_seam[x - 1, possible_y].min()
            energy_among_seam[x, y] = min_energy + energy[x, y]

    return energy_among_seam


def find_seam_with_least_energy(energy):
    seam_mask = np.zeros_like(energy, dtype=np.int8)
    previous_y = np.argmin(energy[-1])
    seam_mask[-1, previous_y] = 1

    for x in range(energy.shape[0] - 2, -1, -1):
        y_max = energy.shape[1] - 1
        y_to_analyze = get_possible_cord(previous_y, energy.shape[1])

        step_directions = (np.argmin(energy[x, y_to_analyze]) -
                           (1 if previous_y else 0))

        previous_y += step_directions
        seam_mask[x, previous_y] = 1

    return seam_mask


def seam_carve(img, command, mask=None):
    energy = count_energy(img, mask)
    axis = command.split(' ')[0] != 'horizontal'
    if axis:
        energy = energy.transpose()

    energy = count_energy_among_seam(energy)
    seam = find_seam_with_least_energy(energy)
    if axis:
        seam = seam.transpose()

    return None, None, seam
