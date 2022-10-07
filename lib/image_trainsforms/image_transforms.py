from PIL import Image


def rotate(img: Image) -> Image:  # todo from scratch
    return img.rotate(90)


def flip(img: Image) -> Image:  # todo from scratch
    return img.transpose(Image.FLIP_LEFT_RIGHT)

