from PIL import Image
from tkinter import simpledialog
from numba import njit
import numpy as np

from lib.image_transforms.image_kernels import bicubic_kernel, lanczos_filter
from lib.singleton_objects import ImageObjectSingleton, UISingleton
from lib.image_managers import ImageViewer
import math




@njit
def padding(image: np.ndarray, padding_size: int) -> np.ndarray: # todo перенеси вместо своего паддинга, я не хочу все твои алгосы рефакторить :3
    height, width = image.shape[:2]
    new_image = np.zeros((height + padding_size * 2, width + padding_size * 2, 3), dtype=np.uint8)
    new_image[padding_size:height + padding_size, padding_size:width + padding_size] = image
    return new_image



class ImgFormatTransformer:
    """
    A class that doing some geometric transformation with image
    """

    @staticmethod
    def rotate():  # TODO: поменять на numpy
        """
        Rotates the image 90 degrees
        : return: None
        """
        ImageObjectSingleton.img = ImageObjectSingleton.img.rotate(90)
        ImageViewer.display_img()

    @staticmethod
    def stash_changes():
        ImageObjectSingleton.img_array = ImageObjectSingleton.default_img
        ImageObjectSingleton.color = "RGB"
        for element in UISingleton.current_elements:
            UISingleton.canvas.delete(element)
        ImageViewer.display_img_array(ImageObjectSingleton.default_img)

    @staticmethod
    def flip() -> None:  # TODO: поменять на numpy
        """
        It mirrors the image horizontally
        : return: None
        """
        ImageObjectSingleton.img = ImageObjectSingleton.img.transpose(Image.FLIP_LEFT_RIGHT)
        ImageViewer.display_img()

    @staticmethod
    def get_pixel(source, col, row):
        r, g, b = source[:, :, 0].astype(float), source[:, :, 1].astype(float), source[:, :, 2].astype(float)
        return r[col][row], g[col][row], b[col][row]

    @staticmethod
    def resize_neighbour():
        img = ImageObjectSingleton.img
        img_array = ImageObjectSingleton.img_array
        factor = simpledialog.askfloat(title="Type a factor value", prompt="Try not to use big values",
                                       parent=UISingleton.ui_main)

        W, H = img.size
        newW = int(W * factor)
        newH = int(H * factor)
        newImage_array = np.array(Image.new('RGB', (newW, newH)))
        r, g, b = newImage_array[:, :, 0].astype(float), newImage_array[:, :, 1].astype(float), newImage_array[:, :,
                                                                                                2].astype(float)
        for col in range(newH):
            for row in range(newW):
                coord = row / factor, col / factor
                p = img.getpixel(coord)
                r[col][row], g[col][row], b[col][row] = p[0], p[1], p[2]

        img_array = (np.dstack((r, g, b))).astype(np.uint8)

        ImageObjectSingleton.img_array = img_array
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def bilinear_resize(cls):

        img = ImageObjectSingleton.img_array

        new_width = int(simpledialog.askstring("Input", "Enter new width:"))
        new_height = int(simpledialog.askstring("Input", "Enter new height:"))

        new_center_x = int(simpledialog.askstring("Input", "Enter center cordinate:"))
        new_center_y = int(simpledialog.askstring("Input", "Enter new center coordinate:"))

        new_img = cls._bilinear_resize(img, new_height, new_width).astype(np.uint8)

        ImageObjectSingleton.img_array = new_img.astype(np.uint8)
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    @njit
    def _bilinear_resize(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        height, width = image.shape[:2]
        image = padding(image, 1)
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                x = j * width / new_width
                y = i * height / new_height
                x1 = math.floor(x)
                y1 = math.floor(y)
                x2 = math.ceil(x)
                y2 = math.ceil(y)
                x_diff = x - x1
                y_diff = y - y1
                new_image[i, j] = (1 - x_diff) * (1 - y_diff) * image[y1, x1] + x_diff * (1 - y_diff) * image[y1, x2] + (
                        1 - x_diff) * y_diff * image[y2, x1] + x_diff * y_diff * image[y2, x2]
        return new_image
            

    @staticmethod
    def mitchell():
        ratio = simpledialog.askfloat(title="Type a factor value", prompt="Try not to use big values",
                                      parent=UISingleton.ui_main)
        B_m = simpledialog.askfloat(title="Type a B_m value", prompt="Try not to use big values", parent=UISingleton.ui_main)
        C_m = simpledialog.askfloat(title="Type a C_m value", prompt="Try not to use big values", parent=UISingleton.ui_main)
        img = ImageObjectSingleton.img_array
        H, W, C = img.shape

        img = padding(img, H, W, C)

        dH = math.floor(H * ratio)
        dW = math.floor(W * ratio)

        dst = np.zeros((dH, dW, 3))

        h = 1 / ratio

        for c in range(C):
            for j in range(dH):
                for i in range(dW):
                    # Getting the coordinates of the
                    # nearby values
                    x, y = i * h + 2, j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    # Considering all nearby 16 values
                    mat_l = np.matrix(
                        [[bicubic_kernel(x1, B_m, C_m), bicubic_kernel(x2, B_m, C_m), bicubic_kernel(x3, B_m, C_m),
                          bicubic_kernel(x4, B_m, C_m)]])
                    mat_m = np.matrix([[img[int(y - y1), int(x - x1), c],
                                        img[int(y - y2), int(x - x1), c],
                                        img[int(y + y3), int(x - x1), c],
                                        img[int(y + y4), int(x - x1), c]],
                                       [img[int(y - y1), int(x - x2), c],
                                        img[int(y - y2), int(x - x2), c],
                                        img[int(y + y3), int(x - x2), c],
                                        img[int(y + y4), int(x - x2), c]],
                                       [img[int(y - y1), int(x + x3), c],
                                        img[int(y - y2), int(x + x3), c],
                                        img[int(y + y3), int(x + x3), c],
                                        img[int(y + y4), int(x + x3), c]],
                                       [img[int(y - y1), int(x + x4), c],
                                        img[int(y - y2), int(x + x4), c],
                                        img[int(y + y3), int(x + x4), c],
                                        img[int(y + y4), int(x + x4), c]]])
                    mat_r = np.matrix(
                        [[bicubic_kernel(y1, B_m, C_m)], [bicubic_kernel(y2, B_m, C_m)], [bicubic_kernel(y3, B_m, C_m)],
                         [bicubic_kernel(y4, B_m, C_m)]])

                    # Here the dot function is used to get the dot
                    # product of 2 matrices
                    dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

        ImageObjectSingleton.img_array = dst.astype(np.uint8)
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
    


    @staticmethod
    def lanczos():
        ratio = simpledialog.askfloat(title="Type a factor value", prompt="Try not to use big values",
                                      parent=UISingleton.ui_main)
        img = ImageObjectSingleton.img_array
        H, W, C = img.shape

        img = padding(img, H, W, C)

        dH = math.floor(H * ratio)
        dW = math.floor(W * ratio)

        dst = np.zeros((dH, dW, 3))

        h = 1 / ratio

        for c in range(C):
            print(c)
            for j in range(dH):
                for i in range(dW):
                    # Getting the coordinates of the
                    # nearby values
                    x, y = i * h + 2, j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    # Considering all nearby 16 values
                    mat_l = np.matrix(
                        [[lanczos_filter(x1), lanczos_filter(x2), lanczos_filter(x3), lanczos_filter(x4)]])
                    mat_m = np.matrix([[img[int(y - y1), int(x - x1), c],
                                        img[int(y - y2), int(x - x1), c],
                                        img[int(y + y3), int(x - x1), c],
                                        img[int(y + y4), int(x - x1), c]],
                                       [img[int(y - y1), int(x - x2), c],
                                        img[int(y - y2), int(x - x2), c],
                                        img[int(y + y3), int(x - x2), c],
                                        img[int(y + y4), int(x - x2), c]],
                                       [img[int(y - y1), int(x + x3), c],
                                        img[int(y - y2), int(x + x3), c],
                                        img[int(y + y3), int(x + x3), c],
                                        img[int(y + y4), int(x + x3), c]],
                                       [img[int(y - y1), int(x + x4), c],
                                        img[int(y - y2), int(x + x4), c],
                                        img[int(y + y3), int(x + x4), c],
                                        img[int(y + y4), int(x + x4), c]]])
                    mat_r = np.matrix(
                        [[lanczos_filter(y1)], [lanczos_filter(y2)], [lanczos_filter(y3)], [lanczos_filter(y4)]])

                    # Here the dot function is used to get the dot
                    # product of 2 matrices
                    dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

        ImageObjectSingleton.img_array = dst.astype(np.uint8)
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
