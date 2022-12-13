from itertools import product
from tkinter import simpledialog

import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros, zeros_like, sort, int8, divide, multiply, pad

from lib.image_managers import ImageViewer
from lib.singleton_objects import ImageObjectSingleton, UISingleton


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center: k_size - center, 0 - center: k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = ravel(image[i: i + k_size, j: j + k_size])
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst


def median_filter(gray_img, mask=3):
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with median filter
    """
    # set image borders
    bd = int(mask / 2)
    # copy image size
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            # get mask according with mask
            kernel = ravel(gray_img[i - bd: i + bd + 1, j - bd: j + bd + 1])
            # calculate mask median
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    return median_img


class ImgFilterTransformer:
    @staticmethod
    def gauss_filter():
        img = ImageObjectSingleton.img_array
        gray = cvtColor(img, COLOR_BGR2GRAY)
        sigma = int(simpledialog.askfloat(title="Type a sigma value", prompt="Sigma",
                                          parent=UISingleton.ui_main))
        gaussian3x3 = gaussian_filter(gray, 3, sigma=sigma)
        ImageObjectSingleton.img_array = gaussian3x3
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def median_filter():
        img = ImageObjectSingleton.img_array
        gray = cvtColor(img, COLOR_BGR2GRAY)
        kernel_radius = int(simpledialog.askfloat(title="Type a kernel_radius value", prompt="Kernel radius",
                                                  parent=UISingleton.ui_main))
        median3x3 = median_filter(gray, kernel_radius)
        ImageObjectSingleton.img_array = median3x3
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def sobel_filter():
        img = ImageObjectSingleton.img_array
        [nx, ny, nz] = np.shape(img)
        r_img, g_img, b_img = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # The following operation will take weights and parameters to convert the color image to grayscale
        gamma = 1.400
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma
        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
        sobel_filtered_image = np.zeros(
            shape=(rows, columns))  # initialization of the output image array (all elements are 0)
        # Now we "sweep" the image in both x and y directions and compute the output
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

        ImageObjectSingleton.img_array = sobel_filtered_image
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
