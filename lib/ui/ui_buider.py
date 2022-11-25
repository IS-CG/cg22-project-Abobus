from functools import partial
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
        cls.add_channels_menu()

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

    @staticmethod
    def add_channels_menu():
        channels_menu = Menu(UISingleton.ui_main, tearoff=0)

        # rgb channels:
        rgb_menu = Menu(UISingleton.ui_main, tearoff=0)
        r_channel = partial(ColorTransformer.edit_channels, channels=[1, 2])
        g_channel = partial(ColorTransformer.edit_channels, channels=[0, 2])
        b_channel = partial(ColorTransformer.edit_channels, channels=[0, 1])
        rg_channel = partial(ColorTransformer.edit_channels, channels=[2])
        rb_channel = partial(ColorTransformer.edit_channels, channels=[1])
        gb_channel = partial(ColorTransformer.edit_channels, channels=[0])
        rgb_menu.add_command(label="r", command=r_channel)
        rgb_menu.add_command(label="g", command=g_channel)
        rgb_menu.add_command(label="b", command=b_channel)
        rgb_menu.add_command(label="rg", command=rg_channel)
        rgb_menu.add_command(label="rb", command=rb_channel)
        rgb_menu.add_command(label="gb", command=gb_channel)
        channels_menu.add_cascade(label='RGB', menu=rgb_menu)

        # hls channels:
        hls_menu = Menu(UISingleton.ui_main, tearoff=0)
        h_channel_1 = partial(ColorTransformer.edit_channels, channels=[1, 2])
        l_channel = partial(ColorTransformer.edit_channels, channels=[0, 2])
        s_channel_1 = partial(ColorTransformer.edit_channels, channels=[0, 1])
        hl_channel = partial(ColorTransformer.edit_channels, channels=[2])
        hs_channel_1 = partial(ColorTransformer.edit_channels, channels=[1])
        ls_channel = partial(ColorTransformer.edit_channels, channels=[0])
        hls_menu.add_command(label="h", command=h_channel_1)
        hls_menu.add_command(label="l", command=l_channel)
        hls_menu.add_command(label="s", command=s_channel_1)
        hls_menu.add_command(label="hl", command=hl_channel)
        hls_menu.add_command(label="hs", command=hs_channel_1)
        hls_menu.add_command(label="ls", command=ls_channel)
        channels_menu.add_cascade(label='HLS', menu=hls_menu)

        # hsv channels:
        hsv_menu = Menu(UISingleton.ui_main, tearoff=0)
        h_channel_2 = partial(ColorTransformer.edit_channels, channels=[1, 2])
        s_channel_2 = partial(ColorTransformer.edit_channels, channels=[0, 2])
        v_channel = partial(ColorTransformer.edit_channels, channels=[0, 1])
        hs_channel_2 = partial(ColorTransformer.edit_channels, channels=[2])
        hv_channel = partial(ColorTransformer.edit_channels, channels=[1])
        sv_channel = partial(ColorTransformer.edit_channels, channels=[0])
        hsv_menu.add_command(label="h", command=h_channel_2)
        hsv_menu.add_command(label="s", command=s_channel_2)
        hsv_menu.add_command(label="v", command=v_channel)
        hsv_menu.add_command(label="hs", command=hs_channel_2)
        hsv_menu.add_command(label="hv", command=hv_channel)
        hsv_menu.add_command(label="sv", command=sv_channel)
        channels_menu.add_cascade(label='HSV', menu=hsv_menu)

        # YCrCb channels:
        YCrCb_menu = Menu(UISingleton.ui_main, tearoff=0)
        y_channel = partial(ColorTransformer.edit_channels, channels=[1, 2])
        cr_channel = partial(ColorTransformer.edit_channels, channels=[0, 2])
        cb_channel = partial(ColorTransformer.edit_channels, channels=[0, 1])
        ycr_channel = partial(ColorTransformer.edit_channels, channels=[2])
        ycb_channel = partial(ColorTransformer.edit_channels, channels=[1])
        crcb_channel = partial(ColorTransformer.edit_channels, channels=[0])
        YCrCb_menu.add_command(label="Y", command=y_channel)
        YCrCb_menu.add_command(label="Cr", command=cr_channel)
        YCrCb_menu.add_command(label="Cb", command=cb_channel)
        YCrCb_menu.add_command(label="YCr", command=ycr_channel)
        YCrCb_menu.add_command(label="YCb", command=ycb_channel)
        YCrCb_menu.add_command(label="CrCb", command=crcb_channel)
        channels_menu.add_cascade(label='YCrCb', menu=YCrCb_menu)

        # YCoCg channels:
        YCoCg_menu = Menu(UISingleton.ui_main, tearoff=0)
        y_channel = partial(ColorTransformer.edit_channels, channels=[1, 2])
        co_channel = partial(ColorTransformer.edit_channels, channels=[0, 2])
        cg_channel = partial(ColorTransformer.edit_channels, channels=[0, 1])
        yco_channel = partial(ColorTransformer.edit_channels, channels=[2])
        ycg_channel = partial(ColorTransformer.edit_channels, channels=[1])
        cocg_channel = partial(ColorTransformer.edit_channels, channels=[0])
        YCoCg_menu.add_command(label="Y", command=y_channel)
        YCoCg_menu.add_command(label="Co", command=co_channel)
        YCoCg_menu.add_command(label="Cg", command=cg_channel)
        YCoCg_menu.add_command(label="YCo", command=yco_channel)
        YCoCg_menu.add_command(label="YCg", command=ycg_channel)
        YCoCg_menu.add_command(label="CoCg", command=cocg_channel)
        channels_menu.add_cascade(label='YCoCg', menu=YCoCg_menu)

        # cmy channels:
        cmy_menu = Menu(UISingleton.ui_main, tearoff=0)
        c_channel = partial(ColorTransformer.edit_channels, channels=[1, 2])
        m_channel = partial(ColorTransformer.edit_channels, channels=[0, 2])
        y_channel = partial(ColorTransformer.edit_channels, channels=[0, 1])
        cm_channel = partial(ColorTransformer.edit_channels, channels=[2])
        cy_channel = partial(ColorTransformer.edit_channels, channels=[1])
        my_channel = partial(ColorTransformer.edit_channels, channels=[0])
        cmy_menu.add_command(label="c", command=c_channel)
        cmy_menu.add_command(label="m", command=m_channel)
        cmy_menu.add_command(label="y", command=y_channel)
        cmy_menu.add_command(label="cm", command=cm_channel)
        cmy_menu.add_command(label="cy", command=cy_channel)
        cmy_menu.add_command(label="my", command=my_channel)
        channels_menu.add_cascade(label='CMY', menu=cmy_menu)

        UISingleton.main_menu.add_cascade(label='Edit channels', menu=channels_menu)

