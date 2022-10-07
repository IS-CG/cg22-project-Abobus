import re

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from loguru import logger

from lib.utils import enforce

mains = tk.Tk()
mains.geometry("1200x900")
mains.bg = "BLUE"
mains.title("Image editor")
img = None


def read_img(verbose=False):
    """
        Reads in a PGM/PPM file by the given name and returns its contents in a new numpy
        ndarray with 8/16-bit elements. Also returns the maximum representable value of a
        pixel (typically 255, 1023, 4095, or 65535).
        """
    filespec = filedialog.askopenfilename(title="open")
    valid_extensions = [".pnm", ".ppm", ".pgm", ".PNM", ".PPM", ".PGM"]
    enforce(isinstance(filespec, str) and len(filespec) >= 5,
            "filespec must be a string of length >= 5, was %r." % (filespec))
    enforce(filespec[-4:] in valid_extensions,
            "file extension must be .pnm, .ppm, or .pgm; was %s." % (filespec[-4:]))
    with open(filespec, "rb") as f:
        buf = f.read()
        regex_pnm_header = b"(^(P[56])\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s)"
        match = re.search(regex_pnm_header, buf)
        if match is not None:
            header, typestr, width, height, maxval = match.groups()
            width, height, maxval = int(width), int(height), int(maxval)
            numch = 3 if typestr == b"P6" else 1
            shape = (height, width, numch) if typestr == b"P6" else (height, width)
            if verbose:
                logger.info("Reading file %s " % (filespec), end='')
                print("(w=%d, h=%d, c=%d, maxval=%d)" % (width, height, numch, maxval))
            dtype = ">u2" if maxval > 255 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width * height * numch, offset=len(header))
            pixels = pixels.reshape(shape).astype(np.uint8 if maxval <= 255 else np.uint16)
    display_img_array(pixels)


def display_img_array(data: np.ndarray) -> None:
    global img
    dispimage = ImageTk.PhotoImage(Image.fromarray(data, 'RGB'))
    img = dispimage
    panel.configure(image=dispimage)
    panel.image = dispimage


def rotate(): # TODO: поменять на numpy
    global img
    img = img.rotate(90)
    display_img_array(img)


def flip(): # TODO: поменять на numpy
    global img
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    display_img_array(img)


def save():
    global img
    imgname = filedialog.asksaveasfilename(title="save", defaultextension=".jpg")
    if imgname:
        img.save(imgname)


if __name__ == "__main__":

    panel = tk.Label(mains, bg="BLACK")
    panel.grid(row=0, column=0, rowspan=12, padx=50, pady=50)

    btnOpen = tk.Button(mains, text='Open', width=25, command=read_img, bg="RED")
    btnOpen.grid(row=0, column=1)

    btnRotate = tk.Button(mains, text='Rotate', width=25, command=rotate, bg="BLUE")
    btnRotate.grid(row=1, column=1)

    btnFlip = tk.Button(mains, text='Flip', width=25, command=flip, bg="PINK")
    btnFlip.grid(row=2, column=1)

    btnSave = tk.Button(mains, text='Save', width=25, command=save, bg="YELLOW")
    btnSave.grid(row=3, column=1)

    mains.mainloop()
