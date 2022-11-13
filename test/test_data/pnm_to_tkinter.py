import os
import tkinter as tk


import cv2
import numpy as np
from PIL import Image


def _photo_image(image: np.ndarray):
    height, width = image.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


root = tk.Tk()

Image.open('sample_640×426.pnm')
array = np.asarray(Image.open('sample_640×426.pnm'))
img = _photo_image(array)

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()
canvas.create_image(20, 20, anchor="nw", image=img)

root.mainloop()