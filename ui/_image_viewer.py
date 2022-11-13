from typing import Dict
import tkinter as tk

from PIL import Image


class _ImageViewer:
    def __init__(self):
        self.mains = tk.Tk()
        self.panel = tk.Label(self.mains, bg="BLACK")
        self.img: Image = None
        self.img_array = None
