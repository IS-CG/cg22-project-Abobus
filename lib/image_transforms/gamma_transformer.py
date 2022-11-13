import numpy as np
import cv2

from lib.singleton_objects import ImageObjectSingleton

from lib.image_managers import ImageViewer


class GammaTransformer:
    @staticmethod
    def correct_gamma(display: bool = True) -> None:
        inv_gamma = 1.0 / ImageObjectSingleton.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        img_array = cv2.LUT(ImageObjectSingleton.img_array, table)
        if display:
            ImageViewer.display_img_array(img_array)

    @classmethod
    def up_gamma(cls) -> None:
        ImageObjectSingleton.gamma += 0.1
        cls.correct_gamma()

    @classmethod
    def gamma_down(cls) -> None:
        ImageObjectSingleton.gamma -= 0.1
        cls.correct_gamma()
