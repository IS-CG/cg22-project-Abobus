from functools import partial

from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from numba import njit
import cv2

from lib.singleton_objects import UISingleton, ImageObjectSingleton


class ImageViewer:

    @staticmethod
    def display_img():
        image = ImageTk.PhotoImage(ImageObjectSingleton.img)
        UISingleton.canvas.itemconfig(UISingleton.img_box, image=image)
        UISingleton.canvas.image = image

    @classmethod
    def auto_image_resizer(cls, e):
        if type(ImageObjectSingleton.img_array) != np.ndarray:
            return None

        img = ImageObjectSingleton.img_array
        h, w = ImageObjectSingleton.img_array.shape[:2]
        if e.width < w or e.height < h:
            new_w = e.width if e.width < w else w
            new_h = e.height if e.height < h else h
            resized_bg = cv2.resize(img, (new_w, new_h))
            cls.display_img_array(resized_bg)

    @classmethod
    def preview_img(cls, data: np.ndarray) -> None:
        top = tk.Toplevel(UISingleton.ui_main)
        top.geometry("700x250")
        top.title("Child Window")
        child_menu = tk.Menu(top)
        top.config(menu=child_menu)
        child_menu.add_command(label='Save Current Img', command=partial(cls.display_img_array, data))
        canvas = tk.Canvas(top, highlightthickness=0, width=600, height=400)
        canvas.pack(fill=tk.BOTH, expand=1)
        img_box = canvas.create_image(0, 0, image=None, anchor='nw')
        img = ImageTk.PhotoImage(Image.fromarray(data))
        canvas.itemconfig(img_box, image=img)
        canvas.image = img

    @classmethod
    def display_img_array(cls, data: np.ndarray) -> None:
        ImageObjectSingleton.img = Image.fromarray(data)
        cls.display_img()
