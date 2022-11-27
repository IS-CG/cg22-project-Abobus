from PIL import Image
import numpy as np

from lib.singleton_objects import ImageObjectSingleton, UISingleton
from lib.image_managers import ImageViewer
import math


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
    def resize_neighbour(): # TODO: сделать конфигурируемые размеры(пока что впадлу менять на partial)
        img = ImageObjectSingleton.img
        img_array = ImageObjectSingleton.img_array
        factor = 2

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

    @staticmethod
    def bilinear_resize():
        """
        `image` is a 2-D numpy array
        `height` and `width` are the desired spatial dimension of the new 2-D array.
        """
        height, width = 800, 600
        image = ImageObjectSingleton.img_array
        img_height, img_width = image.shape[:2]

        resized = np.array(Image.new('RGB', (width, height)))

        x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
        y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

        for i in range(height):
            for j in range(width):
                x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

                x_weight = (x_ratio * j) - x_l
                y_weight = (y_ratio * i) - y_l

                a = image[y_l, x_l]
                b = image[y_l, x_h]
                c = image[y_h, x_l]
                d = image[y_h, x_h]

                pixel = a * (1 - x_weight) * (1 - y_weight) \
                        + b * x_weight * (1 - y_weight) + \
                        c * y_weight * (1 - x_weight) + \
                        d * x_weight * y_weight

                resized[i][j] = pixel

        ImageObjectSingleton.img_array = resized
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
