from PIL import Image, ImageTk
import numpy as np

from lib.singleton_objects import UISingleton, ImageObjectSingleton


class ImageViewer:
    @staticmethod
    def display_img():
        root = UISingleton.ui_main
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        image = ImageTk.PhotoImage(ImageObjectSingleton.img.resize((400, 400)))
        # update the image
        UISingleton.canvas.itemconfig(UISingleton.img_box, image=image)
        # need to keep a reference of the image, otherwise it will be garbage collected
        UISingleton.canvas.image = image

        # UISingleton.canvas.image = dispimage

    @classmethod
    def display_img_array(cls, data: np.ndarray) -> None:
        ImageObjectSingleton.img = Image.fromarray(data)
        cls.display_img()
