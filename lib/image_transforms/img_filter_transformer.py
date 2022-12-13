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


def square_matrix(square):
    """ This function will calculate the value x
       (i.e. blurred pixel value) for each 3 * 3 blur image.
    """
    tot_sum = 0

    # Calculate sum of all the pixels in 3 * 3 matrix
    for i in range(3):
        for j in range(3):
            tot_sum += square[i][j]

    return tot_sum // 9  # return the average of the sum of pixels


# def boxBlur(image):
#     """
#     This function will calculate the blurred
#     image for given n * n image.
#     """
#     square = []  # This will store the 3 * 3 matrix
#     # which will be used to find its blurred pixel
#
#     square_row = []  # This will store one row of a 3 * 3 matrix and
#     # will be appended in square
#
#     blur_row = []  # Here we will store the resulting blurred
#     # pixels possible in one row
#     # and will append this in the blur_img
#
#     blur_img = []  # This is the resulting blurred image
#
#     # number of rows in the given image
#     n_rows = len(image)
#
#     # number of columns in the given image
#     n_col = len(image[0])
#
#     # rp is row pointer and cp is column pointer
#     rp, cp = 0, 0
#
#     # This while loop will be used to
#     # calculate all the blurred pixel in the first row
#     while rp <= n_rows - 3:
#         while cp <= n_col - 3:
#
#             for i in range(rp, rp + 3):
#
#                 for j in range(cp, cp + 3):
#                     # append all the pixels in a row of 3 * 3 matrix
#                     square_row.append(image[i][j])
#
#                 # append the row in the square i.e. 3 * 3 matrix
#                 square.append(square_row)
#                 square_row = []
#
#             # calculate the blurred pixel for given 3 * 3 matrix
#             # i.e. square and append it in blur_row
#             blur_row.append(square_matrix(square))
#             square = []
#
#             # increase the column pointer
#             cp = cp + 1
#
#         # append the blur_row in blur_image
#         blur_img.append(blur_row)
#         blur_row = []
#         rp = rp + 1  # increase row pointer
#         cp = 0  # start column pointer from 0 again
#
#     # Return the resulting pixel matrix
#     return blur_img

def integralImage(img):
    """
    Returns the integral image/summed area table. See here: https://en.wikipedia.org/wiki/Summed_area_table
    :param img:
    :return:
    """
    height = img.shape[0]
    width = img.shape[1]
    int_image = np.zeros((height, width), np.uint64)
    for y in range(height):
        for x in range(width):
            up = 0 if (y - 1 < 0) else int_image.item((y - 1, x))
            left = 0 if (x - 1 < 0) else int_image.item((y, x - 1))
            diagonal = 0 if (x - 1 < 0 or y - 1 < 0) else int_image.item((y - 1, x - 1))
            val = img.item((y, x)) + int(up) + int(left) - int(diagonal)
            int_image.itemset((y, x), val)
    return int_image


def adjustEdges(height, width, point):
    """
    This handles the edge cases if the box's bounds are outside the image range based on current pixel.
    :param height: Height of the image.
    :param width: Width of the image.
    :param point: The current point.
    :return:
    """
    newPoint = [point[0], point[1]]
    if point[0] >= height:
        newPoint[0] = height - 1

    if point[1] >= width:
        newPoint[1] = width - 1
    return tuple(newPoint)


def findArea(int_img, a, b, c, d):
    """
    Finds the area for a particular square using the integral image. See summed area wiki.
    :param int_img: The
    :param a: Top left corner.
    :param b: Top right corner.
    :param c: Bottom left corner.
    :param d: Bottom right corner.
    :return: The integral image.
    """
    height = int_img.shape[0]
    width = int_img.shape[1]
    a = adjustEdges(height, width, a)
    b = adjustEdges(height, width, b)
    c = adjustEdges(height, width, c)
    d = adjustEdges(height, width, d)

    a = 0 if (a[0] < 0 or a[0] >= height) or (a[1] < 0 or a[1] >= width) else int_img.item(a[0], a[1])
    b = 0 if (b[0] < 0 or b[0] >= height) or (b[1] < 0 or b[1] >= width) else int_img.item(b[0], b[1])
    c = 0 if (c[0] < 0 or c[0] >= height) or (c[1] < 0 or c[1] >= width) else int_img.item(c[0], c[1])
    d = 0 if (d[0] < 0 or d[0] >= height) or (d[1] < 0 or d[1] >= width) else int_img.item(d[0], d[1])

    return a + d - b - c


def box_filter(img, filterSize):
    """
    Runs the subsequent box filtering steps. Prints original image, finds integral image, and then outputs final image
    :param img: An image in matrix form.
    :param filterSize: The filter size of the matrix
    :return: A final image written as finalimage.png
    """
    height = img.shape[0]
    width = img.shape[1]
    intImg = integralImage(img)
    finalImg = np.ones((height, width), np.uint64)
    loc = filterSize / 2
    for y in range(height):
        for x in range(width):
            finalImg.itemset((y, x), findArea(intImg, (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc)) / (filterSize ** 2))
    return finalImg


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

    @staticmethod
    def box_blur():
        img = ImageObjectSingleton.img_array
        blurred = box_filter(img, 5)
        ImageObjectSingleton.img_array = np.array(blurred).astype(uint8)
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
