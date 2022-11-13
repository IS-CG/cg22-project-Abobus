import re


import numpy as np
from loguru import logger
from tkinter import filedialog
from PIL import Image

from lib.utils import enforce
from ui.ui_decorators import add_new_img


class ImageReader:
    @staticmethod
    @add_new_img
    def read_img(verbose: bool = False) -> np.ndarray:
        """
        Reads in a PGM/PPM file by the given name and returns its contents in a new numpy
        ndarray with 8/16-bit elements. Also returns the maximum representable value of a
        pixel (typically 255, 1023, 4095, or 65535).
        """
        valid_extensions = [".pnm", ".ppm", ".pgm", ".PNM", ".PPM", ".PGM"]
        filepath = filedialog.askopenfilename(title="open")
        enforce(isinstance(filepath, str) and len(filepath) >= 5,
                "filepath must be a string of length >= 5, was %r." % (filepath))
        enforce(filepath[-4:] in valid_extensions,
                "file extension must be .pnm, .ppm, or .pgm; was %s." % (filepath[-4:]))
        with open(filepath, "rb") as f:
            buf = f.read()
            regex_pnm_header = b"(^(P[56])\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s)"
            match = re.search(regex_pnm_header, buf)
            if match is not None:
                header, typestr, width, height, maxval = match.groups()
                width, height, maxval = int(width), int(height), int(maxval)
                numch = 3 if typestr == b"P6" else 1
                shape = (height, width, numch) if typestr == b"P6" else (height, width)
                if verbose:
                    logger.info("Reading file %s " % (filepath), end='')
                    print("(w=%d, h=%d, c=%d, maxval=%d)" % (width, height, numch, maxval))
                dtype = ">u2" if maxval > 255 else np.uint8
                pixels = np.frombuffer(buf, dtype, count=width * height * numch, offset=len(header))
                pixels = pixels.reshape(shape).astype(np.uint8 if maxval <= 255 else np.uint16)
        return pixels

    @staticmethod
    def save_img(img: Image):
        img_name = filedialog.asksaveasfilename(title="save", defaultextension=".jpg")
        if img_name:
            img.save(img_name)
