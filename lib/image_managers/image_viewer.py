from PIL import Image, ImageTk
import numpy as np
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
        h, w, c = ImageObjectSingleton.img_array.shape
        if e.width < w or e.height < h:
            new_w = e.width if e.width < w else w
            new_h = e.height if e.height < h else h
            resized_bg = cv2.resize(img, (new_w, new_h))
            cls.display_img_array(resized_bg)

    @classmethod
    def display_img_array(cls, data: np.ndarray) -> None:
        ImageObjectSingleton.img = Image.fromarray(data)
        cls.display_img()
