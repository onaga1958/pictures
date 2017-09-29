import numpy as np


def get_derivative_x(matrix):
    return np.concatenate((matrix[1:2] - matrix[:1],
                           matrix[2:] - matrix[:-2],
                           matrix[-1:] - matrix[-2:-1]))

def get_derivative_y(matrix):
    return np.concatenate((matrix[:,1:2] - matrix[:,:1],
                           matrix[:,2:] - matrix[:,:-2],
                           matrix[:,-1:] - matrix[:,-2:-1]), axis=1)


def count_energy(img, mask):
    max_energy = img.shape[0] * img.shape[1] * 256
    brightness_matrix = (0.299*img[:,:,0] +
                         0.587*img[:,:,1] +
                         0.114*img[:,:,2])

    x_derivative = get_derivative_x(brightness_matrix)
    y_derivative = get_derivative_y(brightness_matrix)
    energy = np.sqrt(x_derivative**2 + y_derivative**2)
    if mask is not None:
        energy[mask == 1] += max_energy
        energy[mask == -1] -= max_energy
    return energy

def get_possible_cord(cord, shape):
    return list(range(max(cord - 1, 0), min(cord + 2, shape)))


def count_energy_among_seam(energy):
    energy_among_seam = np.zeros_like(energy)
    energy_among_seam[0] = energy[0]

    for x in range(1, energy.shape[0]):
        for y in range(energy.shape[1]):
            possible_y = get_possible_cord(y, energy.shape[1])
            min_energy = energy_among_seam[x - 1][possible_y].min()
            energy_among_seam[x][y] = min_energy + energy[x][y]
    return energy_among_seam


def find_seam_with_least_energy(energy):
    previous_y = np.argmin(energy[-1])
    seam_mask = np.zeros_like(energy, dtype=np.int8)
    seam_mask[-1][previous_y] = 1

    for x in range(energy.shape[0] - 2, -1, -1):
        y_max = energy.shape[1] - 1
        y_to_analyze = get_possible_cord(previous_y, energy.shape[1])
        step_directions = np.argmin(energy[x][y_to_analyze]) - (1 if previous_y != 0 else 0)
        previous_y += step_directions
        seam_mask[x][previous_y] = 1

    return seam_mask


def remove_seam_with_least_energy(img, energy, mask):
    seam_mask = find_seam_with_least_energy(energy)
    # if mask is not None:
        # mask = mask[seam_mask == 0]

    # shape = list(img.shape)
    # shape[1] -= 1
    # img = img.transpose([2, 0 ,1])
    # img = img[:, seam_mask == 0].transpose().reshape(shape)
    return [img, mask, seam_mask]


def copy_seam_with_least_energy(img, energy, mask):
    seam_mask = find_seam_with_least_energy(energy)
    return [img, mask, seam_mask]


def seam_carve(img, command, mask=None):
    energy = count_energy(img, mask)

    if command == 'horizontal shrink':
        energy = count_energy_among_seam(energy)
        result = remove_seam_with_least_energy(img, energy, mask)

    if command == 'vertical shrink':
        energy = count_energy_among_seam(energy.transpose())
        if mask is not None:
            mask = mask.transpose()
        result = remove_seam_with_least_energy(img.transpose([1, 0, 2]),
                                               energy,
                                               mask)
        # result[0] = result[0].transpose([1, 0, 2])
        # if result[1] is not None:
            # result[1] = result[1].transpose()
        result[2] = result[2].transpose()

    if command == 'horizontal expand':
        energy = count_energy_among_seam(energy)
        result = copy_seam_with_least_energy(img, energy, mask)

    if command == 'vertical expand':
        energy = count_energy_among_seam(energy.transpose())
        if mask is not None:
            mask = mask.transpose()
        result = copy_seam_with_least_energy(img.transpose([1, 0, 2]),
                                             energy,
                                             mask)
        # result[0] = result[0].transpose([1, 0, 2])
        # if result[1] is not None:
            # result[1] = result[1].transpose()
        result[2] = result[2].transpose()

    return result
