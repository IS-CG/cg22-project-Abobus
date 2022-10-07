from typing import Callable

from PIL import ImageTk, Image
import numpy as np

from .ui_singleton import UISingleton


def show_image(data: np.ndarray):
    data.img_array = data
    disp_image = ImageTk.PhotoImage(Image.fromarray(data.img_array, 'RGB'))
    data.img = disp_image
    data.panel.configure(image=disp_image)
    data.panel.image = disp_image


def change_old_img(func: Callable) -> None:
    data = UISingleton.ui_data()
    data.img_array = func(data.img_array)
    show_image(data.img_array)


def add_new_img(func: Callable) -> None:
    data = func()
    show_image(data)
