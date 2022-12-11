import re
from copy import deepcopy
import zlib
import struct
import matplotlib.pyplot as plt

import numpy as np
from loguru import logger
from tkinter import filedialog, messagebox

from lib.utils import enforce
from lib.singleton_objects import ImageObjectSingleton
from lib.image_managers.image_viewer import ImageViewer


def read_chunk(f):
    chunk_length, chunk_type = struct.unpack('>I4s', f.read(8))
    chunk_data = f.read(chunk_length)
    chunk_expected_crc, = struct.unpack('>I', f.read(4))
    chunk_actual_crc = zlib.crc32(chunk_data, zlib.crc32(struct.pack('>4s', chunk_type)))
    if chunk_expected_crc != chunk_actual_crc:
        raise Exception('chunk checksum failed')
    return chunk_type, chunk_data


def PaethPredictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        Pr = a
    elif pb <= pc:
        Pr = b
    else:
        Pr = c
    return Pr


def Recon_a(r, c, Recon, stride, bytesPerPixel):
    return Recon[r * stride + c - bytesPerPixel] if c >= bytesPerPixel else 0


def Recon_b(r, c, Recon, stride):
    return Recon[(r - 1) * stride + c] if r > 0 else 0


def Recon_c(r, c, Recon, stride, bytesPerPixel):
    return Recon[(r - 1) * stride + c - bytesPerPixel] if r > 0 and c >= bytesPerPixel else 0


class ImageReader:
    @staticmethod
    def read_img_pnm(verbose=False):
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
    def read_img_png():
        filespec = filedialog.askopenfilename(title="open")
        with open(filespec, "rb") as f:
            PngSignature = b'\x89PNG\r\n\x1a\n'
            if f.read(len(PngSignature)) != PngSignature:
                raise Exception('Invalid PNG Signature')
            chunks = []
            while True:
                chunk_type, chunk_data = read_chunk(f)
                chunks.append((chunk_type, chunk_data))
                if chunk_type == b'IEND':
                    break
            print([chunk_type for chunk_type, chunk_data in chunks])
            _, IHDR_data = chunks[0]  # IHDR is always first chunk
            width, height, bitd, colort, compm, filterm, interlacem = struct.unpack('>IIBBBBB', IHDR_data)
            if compm != 0:
                raise Exception('invalid compression method')
            if filterm != 0:
                raise Exception('invalid filter method')
            if colort != 6:
                raise Exception('we only support truecolor with alpha')
            if bitd != 8:
                raise Exception('we only support a bit depth of 8')
            if interlacem != 0:
                raise Exception('we only support no interlacing')

            IDAT_data = b''.join(chunk_data for chunk_type, chunk_data in chunks if chunk_type == b'IDAT')
            IDAT_data = zlib.decompress(IDAT_data)

            Recon = []
            bytesPerPixel = 4
            stride = width * bytesPerPixel

            i = 0
            for r in range(height):  # for each scanline
                filter_type = IDAT_data[i]  # first byte of scanline is filter type
                i += 1
                for c in range(stride):  # for each byte in scanline
                    Filt_x = IDAT_data[i]
                    i += 1
                    if filter_type == 0:  # None
                        Recon_x = Filt_x
                    elif filter_type == 1:  # Sub
                        Recon_x = Filt_x + Recon_a(r, c, Recon, stride, bytesPerPixel)
                    elif filter_type == 2:  # Up
                        Recon_x = Filt_x + Recon_b(r, c, Recon, stride)
                    elif filter_type == 3:  # Average
                        Recon_x = Filt_x + (Recon_a(r, c, Recon, stride, bytesPerPixel) + Recon_b(r, c, Recon, stride)) // 2
                    elif filter_type == 4:  # Paeth
                        Recon_x = Filt_x + PaethPredictor(Recon_a(r, c, Recon, stride, bytesPerPixel), Recon_b(r, c, Recon, stride), Recon_c(r, c, Recon, stride, bytesPerPixel))
                    else:
                        raise Exception('unknown filter type: ' + str(filter_type))
                    Recon.append(Recon_x & 0xff)  # truncation to byte
        pixels = np.array(Recon)
        shape = (height, width, 4)
        pixels = pixels.reshape(shape).astype(np.uint8)

        ImageObjectSingleton.img_array = pixels
        ImageObjectSingleton.default_img = deepcopy(pixels)
        ImageViewer.display_img_array(pixels)


    @staticmethod
    def save_img():
        img_name = filedialog.asksaveasfilename(title="save", defaultextension=".jpg")
        if img_name:
            ImageObjectSingleton.img.save(img_name)
