# import numpy as np
#
#
# from .i_dithering_algo import IDitheringAlgo
#
#
# class RandomDitheringAlgo(IDitheringAlgo):
#     @classmethod
#     # @njit
#     def do_dithering(self, img: np.ndarray, color_depth: int) -> np.ndarray:
#         h, w, c = img.shape
#         dithered = np.zeros([h, w])
#
#         for y in range(h):
#             for x in range(w):
#                 rand_values = np.sort(np.random.randint(0, 255, levels - 1))
#                 for i in range(levels - 1):
#                     if image[y][x] < rand_values[i]:
#                         dithered[y][x] = i * inc
#                         break
#                     elif i == (levels - 2):
#                         dithered[y][x] = 255
#                         break
#
#         return dithered
#
