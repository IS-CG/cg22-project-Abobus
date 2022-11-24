from functools import lru_cache, partial
from typing import Tuple

import numpy as np
from tkinter import simpledialog, messagebox
from numba import njit
from PIL import Image
from lib.singleton_objects import ImageObjectSingleton, UISingleton
from lib.image_managers import ImageViewer


class DitheringTransformer:
    _color_depth = 1
    _dithered_img = None

    @classmethod
    def do_dithering(cls, method_name: str, launch_type: str = 'gray'):
        if not(type(ImageObjectSingleton.img_array) is np.ndarray):
            messagebox.showerror('User Input Error', 'You need to apply dithering first!')
            return None
        img_array = ImageObjectSingleton.img_array
        if launch_type == 'gray':
            img_array = cls.get_image(img_array)
        size = img_array.shape
        if len(size) == 2:
            img_array = img_array.reshape((*size, 1))
            
        if method_name == 'ordered':
            new_w = simpledialog.askinteger(title='Write new matrix size dimension',
                                            prompt=f'Default Matrix Kernel is 8',
                                            parent=UISingleton.ui_main)

            dithered_img = cls._ordered_dithering(img_array, new_w)

        elif method_name == 'random':
            dithered_img = cls.random_dithering(img_array)

        elif method_name == 'floyd-steinberg':
            dithered_img = cls._floyd_steinberg_dithering(img_array)

        elif method_name == 'atkinston':
            dithered_img = cls._atkinson_dithering(img_array)
        if len(dithered_img.shape) == 2:
            dithered_img = dithered_img.reshape(-1)
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

    @classmethod
    # @njit
    def random_dithering(cls,
                         image: np.ndarray,
                         size: Tuple[int, int]) -> np.ndarray:
        h, w, c = size
        nc = cls._color_depth
        dithered = np.zeros([h, w])
        levels = cls._color_depth
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
    def _ordered_dithering(cls,
                           image: np.ndarray,
                           d_matrix_kernel: int) -> np.ndarray:

        # @njit
        def calculate_ordered_dithering(dithered: np.ndarray) -> np.ndarray:
            r = 255 / d_matrix_kernel
            for y in range(0, h - 1):
                for x in range(1, w - 1):
                    i = x % d_matrix_kernel
                    j = y % d_matrix_kernel
                    for c in range(channels):
                        value = get_new_val([dithered[y][x][c] + nc * (dither_m[i][j] - 0.5)], nc).item()
                        if value > dither_m[i][j]:
                            dithered[y][x][c] = 255
                        else:
                            dithered[y][x][c] = 0
            return dithered

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

        h, w, channels = image.shape
        nc = cls._color_depth
        dither_m = generate_dither_matrix(d_matrix_kernel)
        dithered = calculate_ordered_dithering(np.copy(image))
        return dithered

    @classmethod
    def _atkinson_dithering(cls,
                            image: np.ndarray,
                            ) -> np.ndarray:

        def atkinson_algo(dithered):
            for y in range(1, h - 3):
                for x in range(1, w - 3):
                    old_pixel = dithered[y][x]
                    new_pixel = get_new_val(old_pixel, nc)
                    quant_error = ((new_pixel - old_pixel) / 8.0).astype(np.uint8)
                    dithered[y, x] = new_pixel

                    # error diffusion
                    dithered[y, x + 1] += quant_error
                    dithered[y, x + 2] += quant_error
                    dithered[y + 1, x] += quant_error
                    dithered[y + 2, x] += quant_error
                    dithered[y + 1, x + 1] += quant_error
                    dithered[y - 1, x - 1] += quant_error
            return dithered

        h, w, c = image.shape
        nc = cls._color_depth
        dithered = np.copy(image)
        dithered = atkinson_algo(dithered)
        carr = np.array(dithered / np.max(dithered, axis=(0, 1)) * 255, dtype=np.uint8)
        return carr

    @classmethod
    def _floyd_steinberg_dithering(cls, arr: np.ndarray) -> np.ndarray:
        @njit
        def calculate_fs(arr):
            for ir in range(h):
                for ic in range(w):
                    old_val = arr[ir, ic].copy()
                    new_val = get_new_val(old_val, nc)
                    arr[ir, ic] = new_val
                    err = old_val - new_val
                    if ic < w - 1:
                        arr[ir, ic + 1] += err * 7 / 16
                    if ir < h - 1:
                        if ic > 0:
                            arr[ir + 1, ic - 1] += err * 3 / 16
                        arr[ir + 1, ic] += err * 5 / 16
                        if ic < w - 1:
                            arr[ir + 1, ic + 1] += err / 16
            return arr

        h, w, c = arr.shape
        nc = 2 ** cls._color_depth
        arr = np.array(arr, dtype=float) / 255
        arr = calculate_fs(arr)
        carr = np.array(arr / np.max(arr, axis=(0, 1)) * 255, dtype=np.uint8)
        return carr

@njit
def get_new_val(old_val, nc):
    return np.array([round(item * (nc - 1)) / (nc - 1) for item in old_val])
