import tkinter as tk
from tkinter import Menu

from lib.image_managers import ImageReader
from lib.singleton_objects import UISingleton
from lib.image_transforms import GammaTransformer, ColorTransformer, ImgFormatTransformer


class UIBuilder:
    @staticmethod
    def build_ui():
        mains = UISingleton.ui_main()

        mains.geometry("1200x900")
        mains.bg = "BLUE"
        mains.title("Image editor")
        panel = tk.Label(mains, bg="BLACK")
        panel.grid(row=0, column=0, rowspan=12, padx=50, pady=50)
        main_menu = Menu(mains)
        mains.config(menu=main_menu)

        file_menu = Menu(main_menu, tearoff=0)
        file_menu.add_command(label="Open", command=ImageReader.read_img)
        file_menu.add_command(label="Save", command=ImageReader.save_img)
        file_menu.add_command(label='Stash Changes', command=ImgFormatTransformer.stash_changes)

        transform_menu = Menu(main_menu, tearoff=0)
        transform_menu.add_command(label="Rotate", command=ImgFormatTransformer.rotate)
        transform_menu.add_command(label="Flip", command=ImgFormatTransformer.flip)

        gamma_menu = Menu(main_menu, tearoff=0)
        gamma_menu.add_command(label='view_new_gamma', command=GammaTransformer.view_new_gamma)
        gamma_menu.add_command(label='change current gamma', command=GammaTransformer.correct_gamma)
        transform_menu.add_cascade(label='Change Gamma', menu=gamma_menu)

        color_menu = Menu(main_menu, tearoff=0)
        color_menu.add_command(label="RGB", command=ColorTransformer.change_to_rgb)
        color_menu.add_command(label="HLS", command=ColorTransformer.change_to_hls)
        color_menu.add_command(label="HSV", command=ColorTransformer.change_to_hsv)
        color_menu.add_command(label="YCrCb_601", command=ColorTransformer.change_to_ycrcb_601)
        color_menu.add_command(label="YCrCb_709", command=ColorTransformer.change_to_ycrcb_709)
        color_menu.add_command(label="YCoCg", command=ColorTransformer.change_to_ycocg)
        color_menu.add_command(label="CMY", command=ColorTransformer.change_to_cmy)

        main_menu.add_cascade(label="File",
                              menu=file_menu)

        transform_menu.add_cascade(label="Change Colors",
                                   menu=color_menu)

        main_menu.add_cascade(label='Transforms',
                              menu=transform_menu)

        UISingleton.panel = panel
        UISingleton.ui_main = mains
        UISingleton.menu = main_menu

