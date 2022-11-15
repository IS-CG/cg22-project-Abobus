import numpy as np
import cv2

from lib.singleton_objects import ImageObjectSingleton

from lib.image_managers import ImageViewer


class GammaTransformer:
    """
    A Class for changing gamma of image

    Args: None
    """
    @staticmethod
    def correct_gamma(display: bool = True) -> None:
        """
        It changes gamma in image array

        Args: display (bool): if True, it will be display image after changing gamma value
        :return: None
        """
        inv_gamma = 1.0 / ImageObjectSingleton.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        img_array = cv2.LUT(ImageObjectSingleton.img_array, table)
        if display:
            ImageViewer.display_img_array(img_array)

    @classmethod
    def up_gamma(cls) -> None:
        """
        It does up gamma value by 0.1

        :return: None
        """
        ImageObjectSingleton.gamma += 0.1
        cls.correct_gamma()

    @classmethod
    def gamma_down(cls) -> None:
        """
        It does decrease gamma value by 0.1

        :return: None
        """
        ImageObjectSingleton.gamma -= 0.1
        cls.correct_gamma()
