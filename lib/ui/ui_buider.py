import tkinter as tk
from tkinter import Menu

from lib.image_managers import ImageReader, ImageViewer
from lib.singleton_objects import UISingleton
from lib.image_transforms import GammaTransformer, ColorTransformer, ImgFormatTransformer


class UIBuilder:
    @staticmethod
    def build_ui():
        mains = UISingleton.ui_main()
        w, h = 1200, 900
        mains.geometry(f"{w}x{h}")
        mains.bg = "BLUE"
        mains.title("Image editor")

        mains.bind('<Configure>', ImageViewer.auto_image_resizer)

        canvas = tk.Canvas(mains, highlightthickness=0, width=600, height=400)
        canvas.pack(fill=tk.BOTH, expand=1)

        img_box = canvas.create_image(0, 0, image=None, anchor='nw')

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

        UISingleton.canvas = canvas
        UISingleton.ui_main = mains
        UISingleton.menu = main_menu
        UISingleton.img_box = img_box

