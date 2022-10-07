from typing import Dict
import tkinter as tk

from PIL import Image

from lib.image_managers import ImageReader
from lib.image_trainsforms import rotate, flip


class _ImageViewer:
    def __int__(self, viewer_conf: Dict[str, str]):
        self.mains = tk.Tk()
        self.panel = tk.Label(self.mains, bg="BLACK")
        self.conf = viewer_conf
        self.img: Image = None
        self.img_array = None
        self._build_ui()

    def _build_ui(self):
        self.mains.geometry(self.viewer_conf['mains']['resolution'])
        self.mains.bg = self.viewer_conf['mains']['bg_color']
        self.mains.title(self.viewer_conf['mains']['title'])

        # todo to auto-config
        self.panel.grid(row=0, column=0, rowspan=12, padx=50, pady=50)

        btnOpen = tk.Button(self.mains, text='Open', width=25, command=ImageReader.read_img, bg="RED")
        btnOpen.grid(row=0, column=1)

        btnRotate = tk.Button(self.mains, text='Rotate', width=25, command=rotate, bg="BLUE")
        btnRotate.grid(row=1, column=1)

        btnFlip = tk.Button(self.mains, text='Flip', width=25, command=flip, bg="PINK")
        btnFlip.grid(row=2, column=1)

        btnSave = tk.Button(self.mains, text='Save', width=25, command=ImageReader.save_img, bg="YELLOW")
        btnSave.grid(row=3, column=1)

    def run_ui(self) -> None:
        """ Starts running ui after call """
        self.mains.mainloop()
