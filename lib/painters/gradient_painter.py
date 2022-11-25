from functools import partial

from PIL import Image
from tkinter import Menu, simpledialog, messagebox
import numpy as np

from .i_painter import IPainter
from lib.singleton_objects import UISingleton,ImageObjectSingleton
from lib.image_managers import ImageViewer


class GradientPainter(IPainter):
    _freq = 4

    @classmethod
    def draw(cls, color: str):
        grad = cls._draw(color)
        ImageObjectSingleton.img_array = grad
        ImageViewer.display_img_array(grad)

    @classmethod
    def get_the_menu(cls):
        popup_menu = Menu(UISingleton.ui_main, tearoff=0)
        popup_menu.add_command(label='Change freq', command=cls.change_freq)
        cascade_menu = Menu(UISingleton.ui_main, tearoff=0)

        cascade_menu.add_command(label='r', command=partial(cls.draw, 'r'))

        cascade_menu.add_command(label='g', command=partial(cls.draw, 'g'))

        cascade_menu.add_command(label='b', command=partial(cls.draw, 'b'))

        cascade_menu.add_command(label='gray', command=partial(cls.draw, 'gray'))

        popup_menu.add_cascade(label='Select Color and Draw', menu=cascade_menu)
        return popup_menu

    @classmethod
    def change_freq(cls) -> None:
        freq = simpledialog.askinteger(title='Write new width',
                                       prompt=f'Current is {cls._width}',
                                       parent=UISingleton.ui_main)
        if freq <= 1:
            messagebox.showerror('User Input Error', 'Freq must be integer 2..6')
        else:
            cls._freq = freq

    @classmethod
    def _draw(cls, color: str) -> np.ndarray:
        color = color.upper()
        w, h = 1000, 500
        r, g, b = 0, 0, 0
        if color == 'GRAY':
            im = Image.new("L", (w, h))
            pixels = im.load()
            for i in range(w):
                if i % cls._freq == 0 and i != 0:
                    r += 1
                for j in range(h):
                    pixels[i, j] = r

        else:
            im = Image.new("RGB", (w, h), (0, 0, 0))
            pixels = im.load()
            if color == 'R':
                for i in range(w):
                    if i % cls._freq == 0 and i != 0:
                        r += 1
                    for j in range(h):
                        pixels[i, j] = r, g, b

            elif color == 'G':
                for i in range(w):
                    if i % cls._freq == 0 and i != 0:
                        g += 1
                    for j in range(h):
                        pixels[i, j] = r, g, b

            elif color == 'B':
                for i in range(w):
                    if i % cls._freq == 0 and i != 0:
                        b += 1
                    for j in range(h):
                        pixels[i, j] = r, g, b

        return np.asarray(im)
