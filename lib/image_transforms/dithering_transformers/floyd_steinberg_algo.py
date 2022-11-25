# import numpy as np
#
# from numba import njit
#
#
# from .i_dithering_algo import IDitheringAlgo
#
#
# class FloydSteinbergDitheringAlgo:
#     @classmethod
#     def do_dithering(cls, arr: np.ndarray) -> np.ndarray:
#         h, w, c = arr.shape
#         nc = 2 ** cls._color_depth
#         arr = np.array(arr, dtype=float) / 255
#         arr = calculate_fs(arr)
#         carr = np.array(arr / np.max(arr, axis=(0, 1)) * 255, dtype=np.uint8)
#
#         return carr
#
#     @classmethod
#     @njit
#     def calculate_fs(cls, arr: np.ndarray) -> np.ndarray:
#         for ir in range(h):
#             for ic in range(w):
#                 old_val = arr[ir, ic].copy()
#                 new_val = get_new_color(old_val, nc)
#                 arr[ir, ic] = new_val
#                 err = old_val - new_val
#                 if ic < w - 1:
#                     arr[ir, ic + 1] += err * 7 / 16
#                 if ir < h - 1:
#                     if ic > 0:
#                         arr[ir + 1, ic - 1] += err * 3 / 16
#                     arr[ir + 1, ic] += err * 5 / 16
#                     if ic < w - 1:
#                         arr[ir + 1, ic + 1] += err / 16
#         return arr
#
#     @staticmethod
#     @njit
#     def get_new_color(old_val, nc):
#         return np.array([round(item * (nc - 1)) / (nc - 1) for item in old_val])
#
