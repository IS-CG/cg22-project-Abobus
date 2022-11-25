from functools import partial

from tkinter import Menu, simpledialog
import cv2


from .i_painter import IPainter
from lib.singleton_objects import UISingleton, ImageObjectSingleton
from lib.image_managers.image_viewer import ImageViewer


class LinePainter(IPainter):
    _width = 5
    _color = (255, 0, 0)
    _old_line_coords = None
    _first_dot_coords = None

    @classmethod
    def draw_line(cls):
        UISingleton.ui_main.bind('<ButtonPress-1>', cls._draw_line)

    @classmethod
    def _draw_line(cls, e=None):
        if e is None:
            coords_list = cls._old_line_coords

        elif cls._first_dot_coords:
            x, y = e.x, e.y
            x1, y1 = cls._first_dot_coords
            coords_list = [(x, y), (x1, y1)]
            cls._old_line_coords = coords_list
            cls._first_dot_coords = None
            UISingleton.ui_main.unbind('<ButtonPress-1>')
        else:
            x, y = e.x, e.y
            coords_list = [(x, y), (x, y)]
            cls._first_dot_coords = x, y

        line_img = cv2.line(ImageObjectSingleton.img_array, *coords_list,
                            thickness=cls._width,
                            color=cls._color,
                            lineType=cv2.LINE_AA)
        ImageObjectSingleton.img_array = line_img

        ImageViewer.display_img_array(line_img)

    @classmethod
    def change_params(cls):

        popup_menu = Menu(UISingleton.ui_main, tearoff=0)
        popup_menu.add_command(label='Change width', command=cls.change_width_func)
        cascade_menu = Menu(UISingleton.ui_main, tearoff=0)

        cascade_menu.add_command(label='red', command=partial(cls._change_color_func, (255, 0, 0)))

        cascade_menu.add_command(label='green', command=partial(cls._change_color_func, (0, 255, 0)))

        cascade_menu.add_command(label='blue', command=partial(cls._change_color_func, (0, 0, 255)))

        popup_menu.add_cascade(label='Change Color', menu=cascade_menu)
        return popup_menu

    @classmethod
    def change_width_func(cls) -> None:
        new_w = simpledialog.askinteger(title='Write new width',
                                        prompt=f'Current is {cls._width}',
                                        parent=UISingleton.ui_main)
        cls._width = new_w
        cls._draw_line()

    @classmethod
    def _change_color_func(cls, color: str):
        cls._color = color
        cls._draw_line()
