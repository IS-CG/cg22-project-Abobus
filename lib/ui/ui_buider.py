from typing import Callable, List

import tkinter as tk
from tkinter import Menu

from lib.image_managers import ImageReader, ImageViewer
from lib.singleton_objects import UISingleton
from lib.image_transforms import GammaTransformer, ColorTransformer, ImgFormatTransformer
from lib.painters import LinePainter


class UIBuilder:
    @classmethod
    def build_ui(cls):
        mains = UISingleton.ui_main()
        w, h = 1200, 900
        mains.geometry(f"{w}x{h}")
        mains.bg = "BLUE"
        mains.title("Image editor")

        mains.bind('<Configure>', ImageViewer.auto_image_resizer)
        UISingleton.ui_main = mains
        cls.build_canvas()
        cls.build_main()
        cls.add_file_menu()
        cls.add_transform_menu()
        cls.add_paint_menu()

    @staticmethod
    def build_main():
        main_menu = Menu(UISingleton.ui_main)
        UISingleton.ui_main.config(menu=main_menu)
        UISingleton.main_menu = main_menu

    @staticmethod
    def build_canvas():
        canvas = tk.Canvas(UISingleton.ui_main, highlightthickness=0, width=600, height=400)
        canvas.pack(fill=tk.BOTH, expand=1)
        img_box = canvas.create_image(0, 0, image=None, anchor='nw')
        UISingleton.canvas = canvas
        UISingleton.img_box = img_box

    @staticmethod
    def add_file_menu():
        file_menu = Menu(UISingleton.ui_main, tearoff=0)
        file_menu.add_command(label="Open", command=ImageReader.read_img)
        file_menu.add_command(label="Save", command=ImageReader.save_img)
        file_menu.add_command(label='Stash Changes', command=ImgFormatTransformer.stash_changes)

        UISingleton.main_menu.add_cascade(label="File",
                                          menu=file_menu)

    @staticmethod
    def add_transform_menu():
        transform_menu = Menu(UISingleton.ui_main, tearoff=0)
        transform_menu.add_command(label="Rotate", command=ImgFormatTransformer.rotate)
        transform_menu.add_command(label="Flip", command=ImgFormatTransformer.flip)

        gamma_menu = Menu(UISingleton.ui_main, tearoff=0)
        gamma_menu.add_command(label='view_new_gamma', command=GammaTransformer.view_new_gamma)
        gamma_menu.add_command(label='change current gamma', command=GammaTransformer.correct_gamma)
        transform_menu.add_cascade(label='Change Gamma', menu=gamma_menu)

        color_menu = Menu(UISingleton.ui_main, tearoff=0)
        color_menu.add_command(label="RGB", command=ColorTransformer.change_to_rgb)
        color_menu.add_command(label="HLS", command=ColorTransformer.change_to_hls)
        color_menu.add_command(label="HSV", command=ColorTransformer.change_to_hsv)
        color_menu.add_command(label="YCrCb_601", command=ColorTransformer.change_to_ycrcb_601)
        color_menu.add_command(label="YCrCb_709", command=ColorTransformer.change_to_ycrcb_709)
        color_menu.add_command(label="YCoCg", command=ColorTransformer.change_to_ycocg)
        color_menu.add_command(label="CMY", command=ColorTransformer.change_to_cmy)

        transform_menu.add_cascade(label="Change Colors",
                                   menu=color_menu)

        UISingleton.main_menu.add_cascade(label='Transforms',
                                          menu=transform_menu)

    @staticmethod
    def add_paint_menu():
        paint_menu = Menu(UISingleton.ui_main, tearoff=0)
        paint_menu.add_command(label="draw_line", command=LinePainter.draw_line)
        paint_menu.add_cascade(label="change_line_params", menu=LinePainter.change_params())
        UISingleton.main_menu.add_cascade(label='Painting',
                                          menu=paint_menu)


