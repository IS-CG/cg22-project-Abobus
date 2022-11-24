import random
import numpy as np
from tkinter import simpledialog, messagebox
from numba import njit

from lib.singleton_objects import ImageObjectSingleton, UISingleton
from lib.image_managers import ImageViewer


D_M = np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]])


class DitheringTransformer:  # todo refactore to strategy pattern
    _color_depth = 1
    _dithered_img = None

    @classmethod
    def do_dithering(cls, method_name: str):
        if not (type(ImageObjectSingleton.img_array) is np.ndarray):
            messagebox.showerror('User Input Error', 'You need to apply dithering first!')
            return None
        img_array = ImageObjectSingleton.img_array

        size = img_array.shape
        if len(size) == 2:
            img_array = img_array.reshape((*size, 1))

        if method_name == 'ordered':

            dithered_img = cls._ordered_dithering(img_array)

        elif method_name == 'random':
            dithered_img = cls._random_dithering(img_array)

        elif method_name == 'floyd-steinberg':
            dithered_img = cls._floyd_steinberg_dithering(img_array)

        elif method_name == 'atkinston':
            dithered_img = cls._atkinson_dithering(img_array)
        if dithered_img.shape[2] == 1:
            dithered_img = dithered_img.reshape(dithered_img.shape[:2])
        cls._dithered_img = dithered_img

    @classmethod
    def change_current_color_depth(cls):
        new_c_d = simpledialog.askinteger(title='Type a num of color depth',
                                          prompt='Color depth must be in range 1..8',
                                          parent=UISingleton.ui_main)
        if not (1 <= new_c_d <= 8):
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
    def _random_dithering(cls,
                          image: np.ndarray) -> np.ndarray:
        @njit
        def calculate_ordered(new_matrix: np.ndarray) -> np.ndarray:
            for y in range(h):
                for x in range(w):
                    pixel = image[y, x].copy()
                    new_pixel = []
                    for p_item in pixel:
                        if random.randint(0, 255) > p_item:
                            new_pixel.append(left_color(p_item, nc))
                        else:
                            new_pixel.append(right_color(p_item, nc))
                    new_matrix[y, x] = np.array(new_pixel)

            return new_matrix

        h, w, channels = image.shape
        nc = cls._color_depth
        new_matrix = np.zeros(image.shape)
        new_matrix = calculate_ordered(new_matrix)
        return new_matrix.astype(np.uint8)
        return dithered

    @classmethod
    def _ordered_dithering(cls,
                           image: np.ndarray) -> np.ndarray:

        @njit
        def calculate_ordered(new_matrix: np.ndarray) -> np.ndarray:
            for y in range(h):
                for x in range(w):
                    pixel = image[y, x].copy()
                    i, j = y % 8, x % 8
                    new_matrix[y, x] = np.array([get_new_color(c, D_M[i][j], nc) for c in pixel], dtype=np.uint8)
            return new_matrix

        h, w, channels = image.shape
        nc = cls._color_depth
        new_matrix = np.zeros(image.shape)
        new_matrix = calculate_ordered(new_matrix)
        return new_matrix.astype(np.uint8)

    @classmethod
    def _atkinson_dithering(cls,
                            image: np.ndarray,
                            ) -> np.ndarray:
        # @njit
        def get_new_color(color: np.ndarray, nc):
            color = color.astype(np.float64)
            c_c = 0
            prev_c = color

            while c_c <= 255:
                c_c += 255 / ((2 ** nc) - 1)
                if prev_c < abs(c_c - color):
                    c_c -= 255 / ((2 ** nc) - 1)
                    break
                prev_c = abs(c_c - color)
            return c_c if c_c > 0 else 0

        # @njit
        def calculate_atkinson(dithered: np.ndarray) -> np.ndarray:
            new_matrix = np.zeros(dithered.shape)
            for y in range(h):
                for x in range(w):
                    pixel = dithered[y, x].copy()
                    new_pixel = np.array([get_new_color(channel, nc) for channel in pixel])
                    error = (new_pixel - pixel) * 0.125
                    for idx in [[x + 1, y], [x + 2, y],
                                [x, y + 1], [x, y + 2]]:
                        try:
                            dithered[idx[1], idx[0]] += error.astype(np.uint8)
                        except IndexError:
                            continue
                    new_matrix[y][x] = new_pixel
            return new_matrix

        h, w, c = image.shape
        nc = cls._color_depth
        dithered = np.copy(image)
        dithered = calculate_atkinson(dithered).astype(np.uint8)
        return dithered

    @classmethod
    def _floyd_steinberg_dithering(cls, arr: np.ndarray) -> np.ndarray:
        @njit
        def get_new_color(old_val, nc):
            return np.array([round(item * (nc - 1)) / (nc - 1) for item in old_val])

        @njit
        def calculate_fs(arr):
            for ir in range(h):
                for ic in range(w):
                    old_val = arr[ir, ic].copy()
                    new_val = get_new_color(old_val, nc)
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


# functions for Random and Ordered Dithering


@njit
def right_color(p_color: np.ndarray, nc):
    n_color = 0
    while n_color < 255 - 10e-5:
        if p_color > n_color + 255 / (2 ** nc - 1):
            n_color += 255 / (2 ** nc - 1 - 10e-5)
        else:
            break
    return n_color + 255 / (2 ** nc - 1)


@njit
def left_color(p_color: np.ndarray, nc):
    n_color = 0
    while n_color < 255 - 10e-5:
        if not p_color < n_color + 255 / (2 ** nc - 1 - 10e-5):
            n_color += 255 / (2 ** nc - 1)
        else:
            break
    return n_color


@njit
def get_new_color(color: np.ndarray, matrix_elem: int, nc: int) -> int:
    l_c = left_color(color, nc)

    if l_c + matrix_elem / 64 * 255 / (2 ** nc - 1) > color:
        return l_c
    else:
        return right_color(color, nc)