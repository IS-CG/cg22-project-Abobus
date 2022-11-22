from functools import lru_cache, partial
from typing import Tuple

import numpy as np
from tkinter import simpledialog, messagebox
from numba import njit

from lib.singleton_objects import ImageObjectSingleton, UISingleton
from lib.image_managers import ImageViewer


class DitheringTransformer:
    _color_depth = 1
    _dithered_img = None

    @classmethod
    def do_dithering(cls, method_name: str):
        if not(type(ImageObjectSingleton.img_array) is np.ndarray):
            messagebox.showerror('User Input Error', 'You need to apply dithering first!')
            return None
        img_array = ImageObjectSingleton.img_array
        gray_scale_img = cls.get_image(img_array)
        size = gray_scale_img.shape[:2]

        if method_name == 'ordered':
            new_w = simpledialog.askinteger(title='Write new matrix size dimension',
                                            prompt=f'Default Matrix Kernel is 8',
                                            parent=UISingleton.ui_main)

            dithered_img = cls._ordered_dithering(gray_scale_img, size, new_w)

        elif method_name == 'random':
            dithered_img = cls.random_dithering(gray_scale_img, size)

        elif method_name == 'floyd-steinberg':
            dithered_img = cls._floyd_steinberg_dithering(gray_scale_img, size)

        elif method_name == 'atkinston':
            dithered_img = cls._atkinson_dithering(gray_scale_img, size)

        cls._dithered_img = dithered_img

    @classmethod
    def change_current_color_depth(cls):
        new_c_d = simpledialog.askinteger(title='Type a num of color depth',
                                          prompt='Color depth must be in range 1..8',
                                          parent=UISingleton.ui_main)
        if not(1 <= new_c_d <= 8):
            messagebox.showerror('User Input Error', 'Color depth must be in range 1..8')
        else:
            cls._color_depth = new_c_d


    @classmethod
    def view_current_dithering(cls):
        if cls._dithered_img is None:
            messagebox.showerror('User Input Error', 'You need to apply dithering first!')
        else:
            ImageViewer.preview_img(cls._dithered_img)

    @staticmethod
    def get_image(img: np.ndarray):
        img_arr = [[(j[0] * 299 / 1000) + (j[1] * 587 / 1000) + (j[2] * 114 / 1000) for j in r] for r in img]
        img_gray = np.array(img_arr)
        return img_gray * (1 / 255)

    @staticmethod
    def int8_truncate(some_int: int) -> int:
        if some_int > 255:
            return 255
        elif some_int < 0:
            return 0
        else:
            return some_int

    @classmethod
    def _floyd_steinberg_dithering(cls,
                                   image: np.ndarray,
                                   size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        dithered = np.copy(image)

        levels = 2 ** cls._color_depth

        for y in range(0, h - 1):
            for x in range(1, w - 1):
                oldpixel = dithered[y][x]
                newpixel = cls.int8_truncate(np.round((levels - 1) * oldpixel / 255) * (255 / (levels - 1)))

                dithered[y][x] = newpixel
                error = oldpixel - newpixel

                dithered[y][x + 1] = cls.int8_truncate(dithered[y][x + 1] + error * 7 / 16.0)
                dithered[y + 1][x - 1] = cls.int8_truncate(dithered[y + 1][x - 1] + error * 3 / 16.0)
                dithered[y + 1][x] = cls.int8_truncate(dithered[y + 1][x] + error * 5 / 16.0)
                dithered[y + 1][x + 1] = cls.int8_truncate(dithered[y + 1][x + 1] + error * 1 / 16.0)

        return dithered

    @classmethod
    def _ordered_dithering(cls,
                           image: np.ndarray,
                           size: Tuple[int, int],
                           d_matrix_kernel: int) -> np.ndarray:
        @lru_cache()
        def generate_dither_matrix(n: int) -> np.ndarray:
            if n == 1:
                return np.array([[0]])
            else:
                first = (n ** 2) * generate_dither_matrix(int(n / 2))
                second = (n ** 2) * generate_dither_matrix(int(n / 2)) + 2
                third = (n ** 2) * generate_dither_matrix(int(n / 2)) + 3
                fourth = (n ** 2) * generate_dither_matrix(int(n / 2)) + 1
                first_col = np.concatenate((first, third), axis=0)
                second_col = np.concatenate((second, fourth), axis=0)
                return (1 / n ** 2) * np.concatenate((first_col, second_col), axis=1)

        h, w = size
        dithered = np.copy(image)
        levels = 2 ** cls._color_depth

        dither_m = generate_dither_matrix(d_matrix_kernel)
        n = np.size(dither_m, axis=0)

        for y in range(0, h - 1):
            for x in range(1, w - 1):
                i = x % n
                j = y % n
                if dithered[y][x] > dither_m[i][j]:
                    dithered[y][x] = cls.int8_truncate(np.round((levels - 1) * 255 / 255) * (255 / (levels - 1)))
                else:
                    dithered[y][x] = 0
        return dithered

    @classmethod
    # @njit
    def random_dithering(cls,
                         image: np.ndarray,
                         size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        dithered = np.zeros([h, w])
        levels = 2 ** cls._color_depth
        inc = 255 / (levels - 1)

        for y in range(h):
            for x in range(w):
                rand_values = np.sort(np.random.randint(0, 255, levels - 1))
                for i in range(levels - 1):
                    if image[y][x] < rand_values[i]:
                        dithered[y][x] = i * inc
                        break
                    elif i == (levels - 2):
                        dithered[y][x] = 255
                        break

        return dithered

    @classmethod
    def _atkinson_dithering(cls, image: np.ndarray,
                            size: Tuple[int, int]
                            ) -> np.ndarray:
        # grab the image dimensions
        h, w = size
        dithered = np.copy(image)
        levels = 2 ** cls._color_depth
        # loop over the image
        for y in range(1, h - 3):
            for x in range(1, w - 3):
                old_pixel = dithered[y][x]
                new_pixel = cls.int8_truncate(np.round((levels - 1) * old_pixel / 255) * (255 / (levels - 1)))
                quant_error = (dithered[y, x] - old_pixel) / 8.0
                dithered[y, x] = new_pixel

                # error diffusion
                dithered[y, x + 1] = cls.int8_truncate(dithered[y, x + 1] + quant_error)
                dithered[y, x + 2] = cls.int8_truncate(dithered[y, x + 2] + quant_error)
                dithered[y + 1, x] = cls.int8_truncate(dithered[y + 1, x] + quant_error)
                dithered[y + 2, x] = cls.int8_truncate(dithered[y + 2, x] + quant_error)
                dithered[y + 1, x + 1] = cls.int8_truncate(dithered[y + 1, x + 1] + quant_error)
                dithered[y - 1, x - 1] = cls.int8_truncate(dithered[y - 1, x - 1] + quant_error)

        return dithered
