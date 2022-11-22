import re
from copy import deepcopy

import numpy as np
from loguru import logger
from tkinter import filedialog, messagebox

from lib.utils import enforce
from lib.singleton_objects import ImageObjectSingleton
from lib.image_managers.image_viewer import ImageViewer


class ImageReader:
    @staticmethod
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
        if not filespec[-4:] in valid_extensions:
            messagebox.showerror('User Input Error',
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
        ImageObjectSingleton.img_array = pixels
        ImageObjectSingleton.default_img = deepcopy(pixels)
        ImageViewer.display_img_array(pixels)

    @staticmethod
    def save_img():
        img_name = filedialog.asksaveasfilename(title="save", defaultextension=".jpg")
        if img_name:
            ImageObjectSingleton.img.save(img_name)
