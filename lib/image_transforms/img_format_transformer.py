from PIL import Image
import numpy as np

from lib.singleton_objects import ImageObjectSingleton
from lib.image_managers import ImageViewer


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
    def resize():
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
                try:
                    coord = col / factor, row / factor
                    p = img.getpixel(coord)
                    r[col][row], g[col][row], b[col][row] = p[0], p[1], p[2]
                except IndexError:
                    kek = 0
        img_array = (np.dstack((r, g, b))).astype(np.uint8)

        ImageObjectSingleton.img_array = img_array
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)
