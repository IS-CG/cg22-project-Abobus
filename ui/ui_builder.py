import tkinter as tk

from lib.image_managers import ImageReader
from lib.image_trainsforms import rotate, flip
from .ui_singleton import UISingleton


class UIBuilder:
    @staticmethod
    def build_and_run():
        conf = UISingleton.ui_conf
        mains = UISingleton.ui_data().mains
        panel = UISingleton.ui_data().panel
        mains.geometry(conf['mains']['resolution'])
        mains.bg = conf['mains']['bg_color']
        mains.title(conf['mains']['title'])

        # todo to auto-config
        panel.grid(row=0, column=0, rowspan=12, padx=50, pady=50)

        btnOpen = tk.Button(mains, text='Open', width=25, command=ImageReader.read_img, bg="RED")
        btnOpen.grid(row=0, column=1)

        btnRotate = tk.Button(mains, text='Rotate', width=25, command=rotate, bg="BLUE")
        btnRotate.grid(row=1, column=1)

        btnFlip = tk.Button(mains, text='Flip', width=25, command=flip, bg="PINK")
        btnFlip.grid(row=2, column=1)

        btnSave = tk.Button(mains, text='Save', width=25, command=ImageReader.save_img, bg="YELLOW")
        btnSave.grid(row=3, column=1)
        """ Starts running ui after call """
        mains.mainloop()
