from itertools import product
from tkinter import simpledialog

import numpy as np
from cv2 import COLOR_RGB2GRAY, cvtColor
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros, zeros_like, sort, int8, divide, multiply
from numba import njit
from lib.image_managers import ImageViewer
from lib.singleton_objects import ImageObjectSingleton, UISingleton


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center: k_size - center, 0 - center: k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i: i + k_size, j: j + k_size])
        image_array[row, :] = window
        row += 1

    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst


def median_filter(gray_img, mask=3):
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with median filter
    """
    bd = int(mask / 2)
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            kernel = ravel(gray_img[i - bd: i + bd + 1, j - bd: j + bd + 1])
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    return median_img


def boxBlur_kernel(square):
    tot_sum = 0
    for i in range(3):
        for j in range(3):
            tot_sum += square[i][j]

    return tot_sum // 9


def boxBlur(image):
    """
    This function will calculate the blurred
    image for given n * n image.
    """
    square = []
    square_row = []
    blur_row = []
    blur_img = []
    n_rows = len(image)
    n_col = len(image[0])

    rp, cp = 0, 0

    while rp <= n_rows - 3:
        while cp <= n_col - 3:

            for i in range(rp, rp + 3):

                for j in range(cp, cp + 3):
                    square_row.append(image[i][j])

                square.append(square_row)
                square_row = []

            blur_row.append(boxBlur_kernel(square))
            square = []

            cp = cp + 1

        blur_img.append(blur_row)
        blur_row = []
        rp = rp + 1
        cp = 0

    return blur_img


def binary_threshold(img, threshold):
    """
    Perform binary thresholding.
    :img: the original grayscale image.
    :threshold: the threshold value.
    """

    f_img = img.flatten()
    binary_mask = (f_img * 255) > threshold
    return np.where(binary_mask, 255, f_img).reshape((img.shape[0], img.shape[1]))


class ImgFilterTransformer:
    @staticmethod
    def gauss_filter():
        img = ImageObjectSingleton.img_array
        gray = cvtColor(img, COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        sigma = int(simpledialog.askfloat(title="Type a sigma value", prompt="Sigma",
                                          parent=UISingleton.ui_main))
        gaussian3x3 = gaussian_filter(gray, 3, sigma=sigma)
        ImageObjectSingleton.img_array = gaussian3x3
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def median_filter():
        img = ImageObjectSingleton.img_array
        gray = cvtColor(img, COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        kernel_radius = int(simpledialog.askfloat(title="Type a kernel_radius value", prompt="Kernel radius",
                                                  parent=UISingleton.ui_main))
        median3x3 = median_filter(gray, kernel_radius)
        ImageObjectSingleton.img_array = median3x3
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def sobel_filter():
        img = ImageObjectSingleton.img_array
        r_img, g_img, b_img = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        gamma = 1.400
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722
        grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma
        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        [rows, columns] = np.shape(grayscale_image)
        sobel_filtered_image = np.zeros(
            shape=(rows, columns))
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))
                gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

        ImageObjectSingleton.img_array = sobel_filtered_image
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def box_blur():
        img = ImageObjectSingleton.img_array
        blurred = boxBlur(img)
        ImageObjectSingleton.img_array = np.array(blurred).astype(uint8)
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def otsu_filter(cls):
        img = ImageObjectSingleton.img_array
        gray = cvtColor(img, COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        otsu = cls.otsu_threshold(gray)
        ImageObjectSingleton.img_array = otsu
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
    
    @staticmethod
    def otsu_threshold(img):
        """
        Perform otsu thresholding.
        :img: the original grayscale image.
        """
        f_img = img.flatten()
        hist, _ = np.histogram(f_img, bins=range(257))
        total = f_img.shape[0]
        sum_bin, w_bin, maxi, level = 0, 0, 0, 0
        sum1 = np.dot(np.arange(0, 256), hist)
        for i in range(0, 256):
            w_bin += hist[i]

            if w_bin == 0:
                continue

            w_f = total - w_bin
            if w_f == 0:
                break

            sum_bin += i * hist[i]
            m_b = sum_bin / w_bin
            m_f = (sum1 - sum_bin) / w_f
            between = w_bin * w_f * (m_b - m_f) ** 2

            if between >= maxi:
                level = i
                maxi = between

        binary_mask = (f_img * 255) > level
        return np.where(binary_mask, 0, f_img).reshape((img.shape[0], img.shape[1]))

    @staticmethod
    def binary_treshold():
        img = ImageObjectSingleton.img_array
        pixel_value = int(simpledialog.askfloat(title="Type a pixel value", prompt="0-255",
                                                parent=UISingleton.ui_main))
        gray = cvtColor(img, COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        final_img = binary_threshold(gray, pixel_value)
        ImageObjectSingleton.img_array = final_img
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    @njit
    def sharpening_filtering(img_array: np.ndarray, coef: float) -> np.ndarray:
        weights = img_array.copy()
        height, width = weights.shape[:2]
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                idxes = [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                min_g = min([weights[i, j, 1] for i, j in idxes]) / 255
                max_g = max([weights[i, j, 1] for i, j in idxes]) / 255
                d_max_g = 1 - max_g
                if (d_max_g > min_g):
                    k_value = 0 if min_g == 0 else d_max_g / max_g
                else:
                    k_value = d_max_g / max_g

                dev_max = -0.125 + coef * (-0.2 - -0.125)
                w = np.sqrt(k_value) * dev_max
                for k in range(3):

                    f_value = sum([w * weights[i, j, k] for i, j in idxes]) / (w * 4 + 1)

                    if f_value > 255 or f_value < 0:
                        f_value = img_array[i, j, k]

                    img_array[i, j, k] = f_value
        return img_array

    @classmethod
    def do_sharpening_filtering(cls):
        coef = int(simpledialog.askinteger(title="Type a pixel value", prompt="0-255",
                                           parent=UISingleton.ui_main))
        img_array = cls.sharpening_filtering(ImageObjectSingleton.img_array, coef)
        ImageObjectSingleton.img_array = img_array
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
