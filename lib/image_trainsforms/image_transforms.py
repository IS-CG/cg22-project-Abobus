from PIL import Image
import numpy as np


from ui import change_old_img


@change_old_img
def rotate(data: np.ndarray) -> np.ndarray:  # todo from scratch
    return np.asarray(Image.fromarray(data).rotate(90))


@change_old_img
def flip(data: np.ndarray) -> np.ndarray:  # todo from scratch
    return np.asarray(Image.fromarray(data).transpose(Image.FLIP_LEFT_RIGHT))
