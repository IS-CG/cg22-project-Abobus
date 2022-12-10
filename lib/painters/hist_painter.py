import numpy as np
from numba import njit

class ImageHistPlotter:
    
    @classmethod
    def draw_img_hist(self,image, bins=256)):
        hist, bins = self.calculate_hist(self.image.ravel(), self.bins, [0, 256])
        ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
    
    @staticmethod
    def automatic_contrast_adjustment(hist, image, bins, range):
        """
        automatic contrast adjustment
        Args:
            image np.ndarray: image to adjust
        Returns: np.ndarray
        """
        a_low = min()
    
    # @njit
    @staticmethod
    def compute_hist(image: np.ndarray,
                    #  bins: int,
                     ignote_part: float) -> np.ndarray:
        """
        compute histogram of image
        Args:
            image: np.ndarray - image to compute histogram
            bins: int - number of bins
            ignote_part: float - float number in range of [0..5]
        Returns: np.ndarray
        """
        
        image_h, image_w, channels  = image.shape
        hist_rgb = np.zeros((channels, 255))
        
        for channel in range(channels):                
            for x in range(image_h):        
                for y in range(image_w):            
                    i = image[x, y]                  
                    hist_rgb[channel, i] += 1
        
        return hist_rgb
        