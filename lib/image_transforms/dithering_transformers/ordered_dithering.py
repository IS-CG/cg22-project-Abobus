# import numpy as np
# from numba import njit
#
# from .i_dithering_algo import IDitheringAlgo
#
# D_M = np.array([
#     [0, 32, 8, 40, 2, 34, 10, 42],
#     [48, 16, 56, 24, 50, 18, 58, 26],
#     [12, 44, 4, 36, 14, 46, 6, 38],
#     [60, 28, 52, 20, 62, 30, 54, 22],
#     [3, 35, 11, 43, 1, 33, 9, 41],
#     [51, 19, 59, 27, 49, 17, 57, 25],
#     [15, 47, 7, 39, 13, 45, 5, 37],
#     [63, 31, 55, 23, 61, 29, 53, 21]])
#
#
# class OrderedDitheringAlgo(IDitheringAlgo):
#
#     @classmethod
#     def do_dithering(cls,
#                      image: np.ndarray,
#                      color_depth: int) -> np.ndarray:
#         new_matrix = np.zeros(image.shape)
#         new_matrix = cls.calculate_ordered(new_matrix, new_matrix, color_depth)
#         return new_matrix.astype(np.uint8)
#
#     @classmethod
#     @njit
#     def calculate_ordered(cls, image: np.ndarray, new_matrix: np.ndarray, nc: int) -> np.ndarray:
#         h, w, channels = image.shape
#         for y in range(h):
#             for x in range(w):
#                 pixel = image[y, x].copy()
#                 i, j = y % 8, x % 8
#                 new_matrix[y, x] = np.array([cls.get_new_color(
#                     c, D_M[i][j], nc) for c in pixel], dtype=np.uint8)
#         return new_matrix
#
#     @classmethod
#     @njit
#     def get_new_color(cls, color: np.ndarray, matrix_elem: int, nc: int) -> int:
#         l_c = cls.left_color(color, nc)
#
#         if l_c + matrix_elem / 64 * 255 / (2 ** nc - 1) > color:
#             return l_c
#         else:
#             return cls.right_color(color, nc)
#
#     @staticmethod
#     @njit
#     def right_color(p_color: np.ndarray, nc):
#         n_color = 0
#         while n_color < 255 - 10e-5:
#             if p_color > n_color + 255 / (2 ** nc - 1):
#                 n_color += 255 / (2 ** nc - 1 - 10e-5)
#             else:
#                 break
#         return n_color + 255 / (2 ** nc - 1)
#
#     @staticmethod
#     @njit
#     def left_color(p_color: np.ndarray, nc):
#         n_color = 0
#         while n_color < 255 - 10e-5:
#             if not p_color < n_color + 255 / (2 ** nc - 1 - 10e-5):
#                 n_color += 255 / (2 ** nc - 1)
#             else:
#                 break
#         return n_color
