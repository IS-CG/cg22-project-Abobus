import re
import cv2

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, Menu
import numpy as np
from loguru import logger
from color_functions import *

from lib.utils import enforce

mains = tk.Tk()
mains.geometry("1200x900")
mains.bg = "BLUE"
mains.title("Image editor")
img = None
img_array = None
COLOR = "RGB"  # by default
GAMMA = 1.0


def read_img(verbose=False):
    global img_array
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
    img_array = pixels
    display_img_array(pixels)


def display_img_array(data: np.ndarray) -> None:
    global img
    img = Image.fromarray(data)
    display_img()


def display_img():
    dispimage = ImageTk.PhotoImage(img)
    panel.configure(image=dispimage)
    panel.image = dispimage


def rotate():  # TODO: поменять на numpy
    global img
    img = img.rotate(90)
    display_img()


def flip():  # TODO: поменять на numpy
    global img
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    display_img()


def save():
    global img
    imgname = filedialog.asksaveasfilename(title="save", defaultextension=".jpg")
    if imgname:
        img.save(imgname)


def correct_gamma(display: bool = True) -> np.ndarray:
    global img_array, GAMMA, mains
    inv_gamma = 1.0 / GAMMA
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    img_array = cv2.LUT(img_array, table)
    if display:
        display_img_array(img_array)


def change_to_rgb(display=True):
    global img_array
    global COLOR
    if COLOR == "HSV":
        h, s, v = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                  2].astype(float) / 255
        r, g, b = np.zeros_like(h), np.zeros_like(s), np.zeros_like(v)
        n, m = h.shape
        for i in range(n):
            for j in range(m):
                if s[i][j] == 0.0:
                    r[i][j], g[i][j], b[i][j] = v[i][j], v[i][j], v[i][j]
                tmp = int(h[i][j] * 6.0)
                f = (h[i][j] * 6.0) - tmp
                p = v[i][j] * (1.0 - s[i][j])
                q = v[i][j] * (1.0 - s[i][j] * f)
                t = v[i][j] * (1.0 - s[i][j] * (1.0 - f))
                tmp = tmp % 6
                if tmp == 0:
                    r[i][j], g[i][j], b[i][j] = v[i][j], t, p
                if tmp == 1:
                    r[i][j], g[i][j], b[i][j] = q, v[i][j], p
                if tmp == 2:
                    r[i][j], g[i][j], b[i][j] = p, v[i][j], t
                if tmp == 3:
                    r[i][j], g[i][j], b[i][j] = p, q, v[i][j]
                if tmp == 4:
                    r[i][j], g[i][j], b[i][j] = t, p, v[i][j]
                if tmp == 5:
                    r[i][j], g[i][j], b[i][j] = v[i][j], p, q
        img_array = (np.dstack((r, g, b)) * 255).astype(np.uint8)
        COLOR = "RGB"
    if COLOR == "HLS":
        h = img_array[..., 0].astype(float) / 255
        l = img_array[..., 1].astype(float) / 255
        s = img_array[..., 2].astype(float) / 255
        n, m = h.shape
        for i in range(n):
            for j in range(m):
                r, g, b = hls_to_rgb(h[i][j], l[i][j], s[i][j])
                h[i][j], l[i][j], s[i][j] = r, g, b
        h, l, s = h, l, s
        img_array = (np.dstack((h, l, s)) * 255).astype(np.uint8)

        COLOR = "RGB"
    if COLOR == "YCrCb_601":
        y, cb, cr = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :,
                                                                                                  2].astype(float)
        r, g, b = np.zeros_like(y), np.zeros_like(cb), np.zeros_like(cr)
        n, m = y.shape
        for i in range(n):
            for j in range(m):
                r[i][j] = y[i][j] + 1.403 * (cr[i][j] - 128)
                g[i][j] = y[i][j] - 0.714 * (cr[i][j] - 128) - 0.344 * (cb[i][j] - 128)
                b[i][j] = y[i][j] + 1.773 * (cb[i][j] - 128)
        img_array = (np.dstack((r, g, b))).astype(np.uint8)

        COLOR = "RGB"

    if COLOR == "CMY":
        c = 255 - img_array[..., 0].astype(float)
        m = 255 - img_array[..., 1].astype(float)
        y = 255 - img_array[..., 2].astype(float)
        img_array = (np.dstack((c, m, y))).astype(np.uint8)
        COLOR = "RGB"

    if display:
        display_img_array(img_array)


def change_to_hls():
    global img_array
    global COLOR
    if COLOR == "RGB":
        r, g, b = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                  2].astype(float) / 255
        maxc = np.max(img_array.astype(float) / 255, -1)
        minc = np.min(img_array.astype(float) / 255, -1)
        l = (minc + maxc) / 2.0
        if np.array_equal(minc, maxc):
            return np.zeros_like(l), l, np.zeros_like(l)
        smask = np.greater(l, 0.5).astype(np.float32)

        s = (1.0 - smask) * ((maxc - minc) / (maxc + minc)) + smask * ((maxc - minc) / (2.001 - maxc - minc))
        rc = (maxc - r) / (maxc - minc + 0.001)
        gc = (maxc - g) / (maxc - minc + 0.001)
        bc = (maxc - b) / (maxc - minc + 0.001)

        rmask = np.equal(r, maxc).astype(np.float32)
        gmask = np.equal(g, maxc).astype(np.float32)
        rgmask = np.logical_or(rmask, gmask).astype(np.float32)

        h = rmask * (bc - gc) + gmask * (2.0 + rc - bc) + (1.0 - rgmask) * (4.0 + gc - rc)
        h = np.remainder(h / 6.0, 1.0)
        img_array = (np.dstack((h, l, s)) * 255).astype(np.uint8)
        COLOR = "HLS"

    if COLOR == "HSV":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2HLS)
        COLOR = "HLS"
    if COLOR == "YCrCb_601":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_YCrCb2RGB), cv2.COLOR_RGB2HLS)
        COLOR = "HLS"
    if COLOR == "CMY":
        change_to_rgb(display=False)
        change_to_hls()
        COLOR = "HLS"
    display_img_array(img_array)


def change_to_hsv():
    global COLOR
    global img_array
    if COLOR == "RGB":
        r, g, b = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                  2].astype(float) / 255
        h, s, v = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
        n, m = r.shape
        for i in range(n):
            for j in range(m):
                maxc = max(r[i][j], g[i][j], b[i][j])
                minc = min(r[i][j], g[i][j], b[i][j])
                rangec = (maxc - minc)
                v_t = maxc
                if minc == maxc:
                    h_t, s_t, v_t = 0.0, 0.0, v_t
                s_t = rangec / maxc
                rc = (maxc - r[i][j]) / rangec
                gc = (maxc - g[i][j]) / rangec
                bc = (maxc - b[i][j]) / rangec
                if r[i][j] == maxc:
                    h_t = bc - gc
                elif g[i][j] == maxc:
                    h_t = 2.0 + rc - bc
                else:
                    h_t = 4.0 + gc - rc
                h_t = (h_t / 6.0) % 1.0
                h[i][j], s[i][j], v[i][j] = h_t, s_t, v_t
        img_array = (np.dstack((h, s, v)) * 255).astype(np.uint8)
        COLOR = "HSV"
    if COLOR == "HLS":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2HSV)
        COLOR = "HSV"
    if COLOR == "YCrCb_601":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_YCrCb2RGB), cv2.COLOR_RGB2HSV)
        COLOR = "HSV"
    if COLOR == "CMY":
        change_to_rgb(display=False)
        change_to_hsv()
        COLOR = "HSV"
    display_img_array(img_array)


def change_to_ycrcb_601():
    global COLOR
    global img_array
    if COLOR == "RGB":
        r, g, b = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :,
                                                                                                  2].astype(float)
        y, cb, cr = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
        n, m = y.shape
        for i in range(n):
            for j in range(m):
                y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j]
                cr[i][j] = (r[i][j] - y[i][j]) * 0.713 + 128
                cb[i][j] = (b[i][j] - y[i][j]) * 0.564 + 128

        img_array = (np.dstack((y, cb, cr))).astype(np.uint8)

        COLOR = "YCrCb_601"
    if COLOR == "HSV":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2YCrCb)
        COLOR = "YCrCb_601"
    if COLOR == "HLS":
        img_array = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2YCrCb)
        COLOR = "YCrCb_601"
    if COLOR == "CMY":
        change_to_rgb(display=False)
        change_to_hsv()
        COLOR = "YCrCb_601"
    display_img_array(img_array)


def change_to_cmy():
    global COLOR
    global img_array
    cmy = None
    if COLOR == "RGB":
        c = 1 - img_array[..., 0].astype(float) / 255
        m = 1 - img_array[..., 1].astype(float) / 255
        y = 1 - img_array[..., 2].astype(float) / 255
        img_array = (np.dstack((c, m, y)) * 255).astype(np.uint8)
        COLOR = "CMY"

    else:
        change_to_rgb(display=False)
        change_to_cmy()

    display_img_array(img_array)


def up_gamma():
    global GAMMA
    GAMMA += 0.1
    correct_gamma()


def gamma_down():
    global GAMMA
    GAMMA += 0.1
    correct_gamma()


if __name__ == "__main__":
    panel = tk.Label(mains, bg="BLACK")
    panel.grid(row=0, column=0, rowspan=12, padx=50, pady=50)

    main_menu = Menu(mains)
    mains.config(menu=main_menu)

    file_menu = Menu(main_menu, tearoff=0)
    file_menu.add_command(label="Open", command=read_img)
    file_menu.add_command(label="Save", command=save)

    transform_menu = Menu(main_menu, tearoff=0)
    transform_menu.add_command(label="Rotate", command=rotate)
    transform_menu.add_command(label="Flip", command=flip)

    gamma_menu = Menu(main_menu, tearoff=0)
    gamma_menu.add_command(label='set up gamma', command=up_gamma)
    gamma_menu.add_command(label='set down gamma', command=gamma_down)
    transform_menu.add_cascade(label='Change Gamma', menu=gamma_menu)

    color_menu = Menu(main_menu, tearoff=0)
    color_menu.add_command(label="RGB", command=change_to_rgb)
    color_menu.add_command(label="HLS", command=change_to_hls)
    color_menu.add_command(label="HSV", command=change_to_hsv)
    color_menu.add_command(label="YCrCb_601", command=change_to_ycrcb_601)
    color_menu.add_command(label="CMY", command=change_to_cmy)

    # color_menu.add_command(label="YCoCg") # TODO: пока хз че это

    main_menu.add_cascade(label="File",
                          menu=file_menu)

    main_menu.add_cascade(label="Colors",
                          menu=color_menu)

    main_menu.add_cascade(label='Transforms',
                          menu=transform_menu)

    mains.mainloop()
