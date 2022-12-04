import numpy as np


def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img

    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]

    return zimg


def bicubic_kernel(x, B=1 / 3., C=1 / 3.):
    if abs(x) < 1:
        return 1 / 6. * ((12 - 9 * B - 6 * C) * abs(x) ** 3 + ((-18 + 12 * B + 6 * C) * abs(x) ** 2 + (6 - 2 * B)))
    elif 1 <= abs(x) < 2:
        return 1 / 6. * (
                (-B - 6 * C) * abs(x) ** 3 + (6 * B + 30 * C) * abs(x) ** 2 + (-12 * B - 48 * C) * abs(x) + (
                8 * B + 24 * C))
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
