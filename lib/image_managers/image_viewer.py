from PIL import Image, ImageTk
import numpy as np

from lib.singleton_objects import UISingleton, ImageObjectSingleton


class ImageViewer:
    @staticmethod
    def display_img():
        dispimage = ImageTk.PhotoImage(ImageObjectSingleton.img)
        UISingleton.panel.configure(image=dispimage)
        UISingleton.panel.image = dispimage

    @classmethod
    def display_img_array(cls, data: np.ndarray) -> None:
        ImageObjectSingleton.img = Image.fromarray(data)
        cls.display_img()
