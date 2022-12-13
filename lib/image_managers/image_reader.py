import re
from copy import deepcopy
from array import array
import zlib
import struct

import cv2
import matplotlib.pyplot as plt

import numpy as np
from loguru import logger
from tkinter import filedialog, messagebox

from numba import njit

from lib.utils import enforce
from lib.singleton_objects import ImageObjectSingleton
from lib.image_managers.image_viewer import ImageViewer


def write_pnm(img_name):
    img = ImageObjectSingleton.img_array
    maxval = np.amax(img)
    height, width = img.shape[0:2]
    ppm_header = f'P6 {width} {height} {maxval}\n'
    with open(img_name, 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        img.tofile(f)


def change_gamma_cv2(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def default_gamma(value: float, gamma: float) -> float:
    return ((value / 255.0) ** gamma) * 255


def srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    else:
        return ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return value * 12.92
    else:
        return 1.055 * (value ** 0.41666) - 0.055


def calculate_new_gamma(img: np.ndarray, inv_gamma: float, last_gamma: float) -> np.ndarray:
    img_row, img_column, channels = img.shape
    r1 = np.zeros((img_row, img_column, channels), dtype=np.float64)
    for i in range(img_row):
        for j in range(img_column):
            if not (last_gamma is None) and last_gamma == 0:
                r1[i, j] = np.array([default_gamma(srgb_to_linear(item),
                                                   gamma=inv_gamma) for item in img[i, j]]).squeeze()
            elif inv_gamma == 0:
                r1[i, j] = np.array([(linear_to_srgb(item / 255)) * 255 for item in img[i, j]]).squeeze()

            else:
                r1[i, j] = np.array([default_gamma(item, gamma=inv_gamma) for item in img[i, j]]).squeeze()
    return r1


def array_scanlines(pixels):
    width, height = pixels.shape[0:2]
    row_bytes = width * 4
    stop = 0
    for y in range(height):
        start = stop
        stop = start + row_bytes
        yield pixels[start:stop]


def write_png(outfile, scanlines, width, height, gamma):
    outfile.write(struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10))
    compression = 2

    interlaced = 0
    write_chunk(outfile, 'IHDR',
                struct.pack("!2I5B", width, height,
                            4 * 8,
                            2, 0, 0, interlaced))

    if gamma is not None:
        write_chunk(outfile, 'gAMA',
                    struct.pack("!L", int(gamma * 100000)))

    if compression is not None:
        compressor = zlib.compressobj(compression)
    else:
        compressor = zlib.compressobj()

    data = array('B')
    for scanline in scanlines:
        data.append(0)
        data.extend(scanline)
        if len(data) > 2 ** 20:
            compressed = compressor.compress(data.tostring())
            if len(compressed):
                write_chunk(outfile, 'IDAT', compressed)
            data = array('B')
    if len(data):
        compressed = compressor.compress(data.tostring())
    else:
        compressed = ''
    flushed = compressor.flush()
    if len(compressed) or len(flushed):
        write_chunk(outfile, 'IDAT', compressed + flushed)

    write_chunk(outfile, 'IEND', '')


def write_chunk(outfile, tag, data):
    outfile.write(struct.pack("!I", len(data)))
    outfile.write(tag)
    outfile.write(data)
    checksum = zlib.crc32(tag)
    checksum = zlib.crc32(data, checksum)
    outfile.write(struct.pack("!i", checksum))


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
            gama_data = b''.join(chunk_data for chunk_type, chunk_data in chunks if chunk_type == b'gAMA')
            gama_value = None
            if gama_data:
                gama_value = struct.unpack("!L", gama_data)[0] / 80
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
                        Recon_x = Filt_x + (
                                Recon_a(r, c, Recon, stride, bytesPerPixel) + Recon_b(r, c, Recon, stride)) // 2
                    elif filter_type == 4:  # Paeth
                        Recon_x = Filt_x + PaethPredictor(Recon_a(r, c, Recon, stride, bytesPerPixel),
                                                          Recon_b(r, c, Recon, stride),
                                                          Recon_c(r, c, Recon, stride, bytesPerPixel))
                    else:
                        raise Exception('unknown filter type: ' + str(filter_type))
                    Recon.append(Recon_x & 0xff)  # truncation to byte
        pixels = np.array(Recon)
        shape = (height, width, 4)
        pixels = pixels.reshape(shape).astype(np.uint8)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)

        if gama_value:
            pixels = calculate_new_gamma(pixels, gama_value, 1)
        pixels = pixels.astype(np.uint8)
        ImageObjectSingleton.img_array = pixels
        ImageObjectSingleton.default_img = deepcopy(pixels)
        ImageViewer.display_img_array(pixels)

    @staticmethod
    def save_pnm():
        img_name = filedialog.asksaveasfilename(title="save", defaultextension=".pnm")
        write_pnm(img_name=img_name)


    @staticmethod
    def save_png():
        img_name = filedialog.asksaveasfilename(title="save", defaultextension=".png")
        if img_name:
            ImageObjectSingleton.img.save(img_name)
            return
        img = ImageObjectSingleton.img_array
        width, height = img.shape[0:2]
        gamma = ImageObjectSingleton.gamma
        with open('img.png', 'wb') as outfile:
            write_png(outfile, array_scanlines(img), width, height, gamma)
