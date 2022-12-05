import numpy as np


def padding(img, H, W, C):
    padded_image = np.zeros((H + 4, W + 4, C))
    padded_image[2:H + 2, 2:W + 2, :C] = img

    # Pad the first/last two col and row
    padded_image[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    padded_image[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    padded_image[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    padded_image[0:2, 2:W + 2, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    padded_image[0:2, 0:2, :C] = img[0, 0, :C]
    padded_image[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    padded_image[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    padded_image[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]

    return padded_image


def bicubic_kernel(x, b=1/3., c=1/3.):
    if abs(x) < 1:
        return 1 / 6. * ((12 - 9 * b - 6 * c) * abs(x) ** 3 + ((-18 + 12 * b + 6 * c) * abs(x) ** 2 + (6 - 2 * b)))
    elif 1 <= abs(x) < 2:
        return 1 / 6. * (
                (-b - 6 * c) * abs(x) ** 3 + (6 * b + 30 * c) * abs(x) ** 2 + (-12 * b - 48 * c) * abs(x) + (
                8 * b + 24 * c))
    else:
        return 0


def sinc_filter(x):
    if x == 0:
        return 1
    x *= np.pi
    return np.sin(x) / x


def lanczos_filter(x):
    if -3.0 <= x < 3.0:
        return sinc_filter(x) * sinc_filter(x / 3)
    else:
        return 0
