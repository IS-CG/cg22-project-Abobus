from PIL import Image


from lib.singleton_objects import ImageObjectSingleton
from lib.image_managers import ImageViewer


class ImgFormatTransformer:
    @staticmethod
    def rotate():  # TODO: поменять на numpy
        ImageObjectSingleton.img = ImageObjectSingleton.img.rotate(90)
        ImageViewer.display_img()

    @staticmethod
    def flip():  # TODO: поменять на numpy
        ImageObjectSingleton.img = ImageObjectSingleton.img.transpose(Image.FLIP_LEFT_RIGHT)
        ImageViewer.display_img()

    @staticmethod
    def stash_changes():
        ImageObjectSingleton.img_array = ImageObjectSingleton.default_img
        ImageViewer.display_img_array(ImageObjectSingleton.default_img)
