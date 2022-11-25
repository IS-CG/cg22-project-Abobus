from tkinter import simpledialog, messagebox
import numpy as np
from numba import njit
from math import floor

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
        if value < 0:
            messagebox.showerror('USER INPUT ERROR', 'GAMMA MUST BE > 0')
        else:
            ImageObjectSingleton.last_gamma = ImageObjectSingleton.gamma
            ImageObjectSingleton.gamma = value  # if value > 0.1 else ImageObjectSingleton.gamma

    @classmethod
    def view_new_gamma(cls, display: bool = True) -> None:
        """
        It changes gamma in image array

        Args: display (bool): if True, it will be display image after changing gamma value
        :return: None
        """
        gamma = ImageObjectSingleton.gamma
        if len(ImageObjectSingleton.img_array.shape) == 2:
            messagebox.showerror('USER INPUT ERROR', 'MUST BE RGB IMAGE')
        img = cls.calculate_new_gamma(img=ImageObjectSingleton.img_array,
                                      inv_gamma=gamma,
                                      last_gamma=ImageObjectSingleton.last_gamma).astype("uint8")

        if display:
            ImageViewer.display_img_array(img)

    @staticmethod
    @njit
    def calculate_new_gamma(img: np.ndarray, inv_gamma: float, last_gamma: float) -> np.ndarray:
        img_row, img_column, channels = img.shape
        r1 = np.zeros((img_row, img_column, channels), dtype=np.float64)
        for i in range(img_row):
            for j in range(img_column):
                if not(last_gamma is None) and last_gamma == 0:
                    r1[i, j] = np.array([default_gamma(srgb_to_linear(item),
                                                       gamma=inv_gamma) for item in img[i, j]])
                elif inv_gamma == 0:
                    r1[i, j] = np.array([(linear_to_srgb(item / 255)) * 255 for item in img[i, j]])

                else:
                    r1[i, j] = np.array([default_gamma(item, gamma=inv_gamma) for item in img[i, j]])
        return r1


@njit
def linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return value * 12.92
    else:
        return 1.055 * (value ** 0.41666) - 0.055


@njit
def srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    else:
        return ((value + 0.055) / 1.055) ** 2.4


@njit
def default_gamma(value: float, gamma: float) -> float:
    return ((value / 255.0) ** gamma) * 255
