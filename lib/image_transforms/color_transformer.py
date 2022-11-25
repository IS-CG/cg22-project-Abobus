import numpy as np

from lib.image_managers import ImageViewer
from lib.singleton_objects import ImageObjectSingleton


ONE_THIRD = 1.0 / 3.0
ONE_SIXTH = 1.0 / 6.0
TWO_THIRD = 2.0 / 3.0


class ColorTransformer:
    """
    A Class for transfer image from one color space to another

    Args: None
    """
    @classmethod
    def change_to_rgb(cls,
                      display=True):
        """
        It takes image array from other color space and changes to rgb

        Args: display (bool): if True, it will be display image into UI after changing color space
        : return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color
        if color == "HSV":
            h, s, v = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                      2].astype(float) / 255
            r, g, b = np.zeros_like(h), np.zeros_like(s), np.zeros_like(v)
            n, m = h.shape
            for i in range(n):
                for j in range(m):
                    if s[i][j] == 0.0:
                        r[i][j], g[i][j], b[i][j] = v[i][j], v[i][j], v[i][j]
                    tmp = int(h[i][j] * 6.0)
                    f = (h[i][j] * 6.0) - tmp
                    p = v[i][j] * (1.0 - s[i][j])
                    q = v[i][j] * (1.0 - s[i][j] * f)
                    t = v[i][j] * (1.0 - s[i][j] * (1.0 - f))
                    tmp = tmp % 6
                    if tmp == 0:
                        r[i][j], g[i][j], b[i][j] = v[i][j], t, p
                    if tmp == 1:
                        r[i][j], g[i][j], b[i][j] = q, v[i][j], p
                    if tmp == 2:
                        r[i][j], g[i][j], b[i][j] = p, v[i][j], t
                    if tmp == 3:
                        r[i][j], g[i][j], b[i][j] = p, q, v[i][j]
                    if tmp == 4:
                        r[i][j], g[i][j], b[i][j] = t, p, v[i][j]
                    if tmp == 5:
                        r[i][j], g[i][j], b[i][j] = v[i][j], p, q
            img_array = (np.dstack((r, g, b)) * 255).astype(np.uint8)
            color = "RGB"
        if color == "HLS":
            h = img_array[..., 0].astype(float) / 255
            l = img_array[..., 1].astype(float) / 255
            s = img_array[..., 2].astype(float) / 255
            n, m = h.shape
            for i in range(n):
                for j in range(m):
                    r, g, b = cls.hls_to_rgb(h[i][j], l[i][j], s[i][j])
                    h[i][j], l[i][j], s[i][j] = r, g, b
            h, l, s = h, l, s
            img_array = (np.dstack((h, l, s)) * 255).astype(np.uint8)

            color = "RGB"
        if color == "YCrCb_601":
            y, cb, cr = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :,
                                                                                            2].astype(float)
            r, g, b = np.zeros_like(y), np.zeros_like(cb), np.zeros_like(cr)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    r[i][j] = y[i][j] + 1.403 * (cr[i][j] - 128)
                    g[i][j] = y[i][j] - 0.714 * (cr[i][j] - 128) - 0.344 * (cb[i][j] - 128)
                    b[i][j] = y[i][j] + 1.773 * (cb[i][j] - 128)
            img_array = (np.dstack((r, g, b))).astype(np.uint8)

            color = "RGB"

        if color == "YCrCb_709":
            y, cb, cr = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :,
                                                                                            2].astype(float)
            r, g, b = np.zeros_like(y), np.zeros_like(cb), np.zeros_like(cr)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    r[i][j] = y[i][j] + 1.5748 * (cr[i][j] - 128)
                    g[i][j] = y[i][j] - 0.635 * (cr[i][j] - 128) - 0.1873 * (cb[i][j] - 128)
                    b[i][j] = y[i][j] + 1.8556 * (cb[i][j] - 128)
            img_array = (np.dstack((r, g, b))).astype(np.uint8)

            color = "RGB"

        if color == "YCoCg":
            y, co, cg = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                        2].astype(
                float) / 255
            r, g, b = np.zeros_like(y), np.zeros_like(co), np.zeros_like(cg)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    r[i][j] = y[i][j] + co[i][j] - cg[i][j]
                    g[i][j] = y[i][j] + cg[i][j]
                    b[i][j] = y[i][j] - co[i][j] - cg[i][j]
            img_array = (np.dstack((r, g, b)) * 255).astype(np.uint8)

            color = "RGB"

        if color == "CMY":
            c = 255 - img_array[..., 0].astype(float)
            m = 255 - img_array[..., 1].astype(float)
            y = 255 - img_array[..., 2].astype(float)
            img_array = (np.dstack((c, m, y))).astype(np.uint8)
            color = "RGB"

        ImageObjectSingleton.img_array = img_array
        ImageObjectSingleton.color = color

        if display:
            ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_hls(cls):
        """
        It takes image array from other color space and changes to hls

        Args: None
        : return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color

        if color == "RGB":
            r, g, b = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                      2].astype(float) / 255
            maxc = np.max(img_array.astype(float) / 255, -1)
            minc = np.min(img_array.astype(float) / 255, -1)
            l = (minc + maxc) / 2.0
            if np.array_equal(minc, maxc):
                return np.zeros_like(l), l, np.zeros_like(l)
            smask = np.greater(l, 0.5).astype(np.float32)

            s = (1.0 - smask) * ((maxc - minc) / (maxc + minc)) + smask * ((maxc - minc) / (2.001 - maxc - minc))
            rc = (maxc - r) / (maxc - minc + 0.001)
            gc = (maxc - g) / (maxc - minc + 0.001)
            bc = (maxc - b) / (maxc - minc + 0.001)

            rmask = np.equal(r, maxc).astype(np.float32)
            gmask = np.equal(g, maxc).astype(np.float32)
            rgmask = np.logical_or(rmask, gmask).astype(np.float32)

            h = rmask * (bc - gc) + gmask * (2.0 + rc - bc) + (1.0 - rgmask) * (4.0 + gc - rc)
            h = np.remainder(h / 6.0, 1.0)
            img_array = (np.dstack((h, l, s)) * 255).astype(np.uint8)
            ImageObjectSingleton.color = "HLS"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_hls()
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_hsv(cls):
        """
        It takes image array from other color space and changes to hsv

        Args: None
        :return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color

        if color == "RGB":
            r, g, b = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                      2].astype(float) / 255
            h, s, v = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
            n, m = r.shape
            for i in range(n):
                for j in range(m):
                    maxc = max(r[i][j], g[i][j], b[i][j])
                    minc = min(r[i][j], g[i][j], b[i][j])
                    rangec = (maxc - minc)
                    v_t = maxc
                    if minc == maxc:
                        h_t, s_t, v_t = 0.0, 0.0, v_t
                    s_t = rangec / maxc
                    rc = (maxc - r[i][j]) / rangec
                    gc = (maxc - g[i][j]) / rangec
                    bc = (maxc - b[i][j]) / rangec
                    if r[i][j] == maxc:
                        h_t = bc - gc
                    elif g[i][j] == maxc:
                        h_t = 2.0 + rc - bc
                    else:
                        h_t = 4.0 + gc - rc
                    h_t = (h_t / 6.0) % 1.0
                    h[i][j], s[i][j], v[i][j] = h_t, s_t, v_t
            img_array = (np.dstack((h, s, v)) * 255).astype(np.uint8)
            ImageObjectSingleton.color = "HSV"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_hsv()

        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_ycrcb_601(cls):
        """
        It takes image array from other color space and changes to YCbCr_601

        Args: None
        : return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color

        if color == "RGB":
            r, g, b = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :,
                                                                                          2].astype(float)
            y, cb, cr = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j]
                    cr[i][j] = (r[i][j] - y[i][j]) * 0.713 + 128
                    cb[i][j] = (b[i][j] - y[i][j]) * 0.564 + 128

            img_array = (np.dstack((y, cb, cr))).astype(np.uint8)
            ImageObjectSingleton.color = "YCrCb_601"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_ycrcb_601()
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_ycrcb_709(cls):
        """
        It takes image array from other color space and changes to YCbCr_709

        Args: None
        :return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color

        if color == "RGB":
            r, g, b = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :, 2].astype(float)
            y, cb, cr = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    y[i][j] = 0.2126 * r[i][j] + 0.7152 * g[i][j] + 0.0722 * b[i][j]
                    cr[i][j] = (r[i][j] - y[i][j]) * 0.635 + 128
                    cb[i][j] = (b[i][j] - y[i][j]) * 0.5389 + 128

            img_array = (np.dstack((y, cb, cr))).astype(np.uint8)
            ImageObjectSingleton.color = "YCrCb_709"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_ycrcb_709()

        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_ycocg(cls):
        """
        It takes image array from other color space and changes to YCoCg

        Args: None
        : return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color

        if color == "RGB":
            r, g, b = img_array[:, :, 0].astype(float) / 255, img_array[:, :, 1].astype(float) / 255, img_array[:, :,
                                                                                                      2].astype(float) / 255
            y, co, cg = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
            n, m = y.shape
            for i in range(n):
                for j in range(m):
                    y[i][j] = 0.25 * r[i][j] + 0.5 * g[i][j] + 0.25 * b[i][j]
                    co[i][j] = 0.5 * r[i][j] - 0.5 * b[i][j]
                    cg[i][j] = -0.25 * r[i][j] + 0.5 * g[i][j] - 0.25 * b[i][j]
            img_array = (np.dstack((y, co, cg)) * 255).astype(np.uint8)
            ImageObjectSingleton.color = "YCoCg"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_ycocg()

        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @classmethod
    def change_to_cmy(cls):
        """
        It takes image array from other color space and changes to CMY

        Args: None
        : return: None
        """
        img_array = ImageObjectSingleton.img_array
        color = ImageObjectSingleton.color
        if color == "RGB":
            c = 1 - img_array[..., 0].astype(float) / 255
            m = 1 - img_array[..., 1].astype(float) / 255
            y = 1 - img_array[..., 2].astype(float) / 255
            img_array = (np.dstack((c, m, y)) * 255).astype(np.uint8)
            ImageObjectSingleton.color = "CMY"
            ImageObjectSingleton.img_array = img_array
        else:
            cls.change_to_rgb(display=False)
            cls.change_to_cmy()

        ImageViewer.display_img_array(ImageObjectSingleton.img_array)

    @staticmethod
    def hls_to_rgb(h, l, s):
        """
        Args: h -  hue, s - saturation, s - lightness
        : return: None
        """
        def _v(m1, m2, hue):
            hue = hue % 1.0
            if hue < ONE_SIXTH:
                return m1 + (m2 - m1) * hue * 6.0
            if hue < 0.5:
                return m2
            if hue < TWO_THIRD:
                return m1 + (m2 - m1) * (TWO_THIRD - hue) * 6.0
            return m1

        if s == 0.0:
            return l, l, l
        if l <= 0.5:
            m2 = l * (1.0 + s)
        else:
            m2 = l + s - (l * s)
        m1 = 2.0 * l - m2
        return _v(m1, m2, h + ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h - ONE_THIRD)

    @classmethod
    def edit_channels(cls, channels):
        img_array = ImageObjectSingleton.img_array
        for channel in channels:
            img_array[:, :, channel] *= 0

        ImageObjectSingleton.img_array = img_array
        ImageViewer.display_img_array(ImageObjectSingleton.img_array)