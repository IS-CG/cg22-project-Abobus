import io
from typing import Optional
from tkinter import simpledialog

import numpy as np
import cv2
from numba import njit
import matplotlib.pyplot as plt

from lib.singleton_objects import ImageSingleton
from lib.image_managers import ImageViewer


class ImageHistPlotter:
    """ Class for plotting image histogram """
    
    _ignore_range = 0
    _plot_bins = 90

    @classmethod
    def change_ignore_range(cls):
        """ Changes ignore range for histogram """
        simpledialog.askfloat('Ignore range',
                              'Enter new ignore range part - a value [0..0.5]',
                              initialvalue=cls._ignore_range)
        
    @classmethod
    def create_hist_plot(cls, color: str='black', channel: int=0) -> None:
        """ Creates histogram plot for image
        Args:
            color (str, optional): histogram color
            channel (int, optional): channel to calculate histogram. Defaults to 0.
        """        
        img_array = ImageSingleton.img_array.copy()
        
        if channel == -1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        hist = cls.compute_hist_for_channel(img_array)
        cumsum = hist.cumsum()
        hist_array = cls.get_hist_array(img_array, color=color,
                                        bins=cls._plot_bins, cumsum=cumsum)
        ImageViewer.preview_img(hist_array)

    @staticmethod
    def get_hist_array(img_channel: np.ndarray,
                       color: str='black',
                       bins: int=90,
                       cumsum: np.ndarray=None) -> np.ndarray:
        """ Returns histogram plot as np.ndarray

        Args:
            img_channel (np.ndarray): channel to calculate histogram. Defaults to 0.
            color (str, optional): histogram color
            bins (int, optional): bins for hist plotter. Defaults to 90.
            cumsum (np.ndarray, optional): cumsum to plot. Defaults to None.
        Returns:
            np.ndarray : histogram plot as np.ndarray
        """
        io_buf = io.BytesIO()

        fig, ax = plt.subplots(1, 2)

        ax[0].hist(img_channel.flatten(), bins,
                density = False, 
                histtype ='bar',
                color = color,
        )
        ax[0].set_title('Histogram')

        ax[1].plot(cumsum)
        ax[1].set_title('Cumulative Histogram')

        plt.legend(prop ={'size': 10})
        plt.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        return img_arr

    @classmethod
    def apply_contast_correction(cls):
        img_array = ImageSingleton.img_array.copy()
        img_array = cls.auto_contrast_correction(img_array, cls._ignore_range)
        ImageSingleton.img_array = img_array
        ImageViewer.display_img_array(img_array)

    @staticmethod
    def image_channel_from_hist(hist, img_channel):
        cs = hist.cumsum()
        cs = ((cs - cs.min()) * 255) / (cs.max() - cs.min())
        cs = cs.astype('uint8')
        img_channel = cs[img_channel.flatten()].reshape(img_channel.shape)
        return img_channel
    
    @staticmethod
    def auto_contrast_correction(hist: np.ndarray, ignore_range: Optional[float]=None) -> np.ndarray:
        if ignore_range:
            persentile_min, persentile_max = int(ignore_range * 100), int((1 - ignore_range) * 100)
            q_low, q_high = np.percentile(hist, [persentile_min, persentile_max])
            
            a_min, a_max = min(hist), max(hist)
            a_low = min([hist[i] for i in range(len(hist)) if hist[i] >= q_low])
            a_high = max([hist[i] for i in range(len(hist)) if hist[i] <= q_high])
            
            def formula(pixel: int) -> int:
                if pixel < a_low:
                    value =  a_min
                elif a_low <= pixel <= a_high:
                    value = a_min + (pixel - a_low) * (a_max - a_min) / (a_high - a_low)
                elif pixel > a_high:
                    value = a_max
                return value
            
            hist = np.vectorize(formula)(hist).astype(np.uint8)
        return hist
    
    @staticmethod
    @njit
    def compute_hist_for_channel(
                    image: np.ndarray,
                     ) -> np.ndarray:
        """
        compute histogram of image
        Args:
            image: np.ndarray - image to compute histogram
        Returns: np.ndarray
        """

        image_h, image_w  = image.shape
        hist = np.zeros((256))
        
        for x in range(image_h):
            for y in range(image_w):
                i = image[x, y]               
                hist[i] += 1
        
        return hist
