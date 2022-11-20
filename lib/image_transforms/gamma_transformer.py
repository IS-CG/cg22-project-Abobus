from tkinter import simpledialog
import numpy as np
from numba import njit


from lib.singleton_objects import ImageObjectSingleton, UISingleton

from lib.image_managers import ImageViewer


class GammaTransformer:
    """
    A Class for changing gamma of image

    Args: None
    """

    @classmethod
    def correct_gamma(cls) -> None:
        """
        Corrects gamma by user input
        : return: None
        """

        value = simpledialog.askfloat(f'Current gamma value is {ImageObjectSingleton.gamma}',
                                      UISingleton.main_menu,
                                      minvalue=0)

        ImageObjectSingleton.gamma = value

    @classmethod
    def view_new_gamma(cls, display: bool = True) -> None:
        """
        It changes gamma in image array

        Args: display (bool): if True, it will be display image after changing gamma value
        :return: None
        """
        gamma = 0.1 if ImageObjectSingleton.gamma == 0 else ImageObjectSingleton.gamma
        img = cls.calculate_new_gamma(img=ImageObjectSingleton.img_array, inv_gamma=gamma).astype("uint8")

        if display:
            ImageViewer.display_img_array(img)

    @staticmethod
    @njit
    def calculate_new_gamma(img: np.ndarray, inv_gamma: float) -> np.ndarray:
        img_row, img_column, channels = img.shape

        r1 = np.zeros((img_row, img_column, channels), dtype=np.int64)
        for i in range(img_row):
            for j in range(img_column):
                r1[i, j, :] = ((img[i, j] / 255.0) ** inv_gamma) * 255
        return r1

