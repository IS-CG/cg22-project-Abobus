from itertools import product
from tkinter import simpledialog

import numpy as np
import cv2
from cv2 import COLOR_RGB2GRAY, cvtColor
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros, zeros_like, sort, int8, divide, multiply
from numba import njit
from lib.image_managers import ImageViewer
from lib.singleton_objects import ImageObjectSingleton, UISingleton
import math


def gaussian_kernel(mask_size, sigma, twoDimensional=True):
    """
    Creates a gaussian kernel with given sigma and size, 3rd argument is for choose the kernel as 1d or 2d
    """
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(mask_size-1)/2)**2+(y-(mask_size-1)/2)**2))/(2*sigma**2)),
                                 (mask_size, mask_size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(mask_size-1)/2)**2) / (2*sigma**2)), (mask_size,))
    return kernel / np.sum(kernel)


def gaussian_filter(gray_img, mask=3, sigma=1):
    g_kernel_x = gaussian_kernel(mask,sigma, False)
    g_kernel_y = g_kernel_x.reshape(-1,1)
    dst = convolve(gray_img, g_kernel_x)
    dst = convolve(dst, g_kernel_y)
    return dst


def convolve(img: np.array, kernel: np.array) -> np.array:
    """ Applies a 2d convolution """
    def add_padding_to_image(img: np.array, kernel_size: int) -> np.array:
        def get_padding_width_per_side(kernel_size: int) -> int:
            return kernel_size // 2

        padding_width = get_padding_width_per_side(kernel_size)
        img_with_padding = np.zeros(shape=(
            img.shape[0] + padding_width * 2,
            img.shape[1] + padding_width * 2
        ))
        img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img

        return img_with_padding

    return cv2.filter2D(img, -1, kernel)

    def calculate_target_size(img_size: int, kernel_size: int) -> int:
        num_pixels = 0
        for i in range(img_size):
            added = i + kernel_size
            if added <= img_size:
                num_pixels += 1
        return num_pixels

    pad_img = add_padding_to_image(img, kernel_size=kernel.shape[0])
    tgt_size = calculate_target_size(
        img_size=max(pad_img.shape[0], pad_img.shape[1]),
        kernel_size=kernel.shape[0]
    )
    k = kernel.shape[0]
    convolved_img = np.zeros(shape=(tgt_size, tgt_size))

    for i in range(tgt_size):
        for j in range(tgt_size):
            mat = pad_img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img

    
    

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
        mask_size = 3 * sigma
        gaussian3x3 = gaussian_filter(gray, mask=mask_size, sigma=sigma)
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
