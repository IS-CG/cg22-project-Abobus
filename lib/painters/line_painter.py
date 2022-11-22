from tkinter import Menu, simpledialog

from .i_painter import IPainter
from lib.singleton_objects import UISingleton
from lib.ui.ui_builders import UICascadeGenerator


class LinePainter(IPainter):
    _width = 5
    _color = "red"
    _all_possible_colors = ["white", "black", "red", "green",
                            "blue", "cyan", "yellow", "magenta"]
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
            coords_list = [x, y, x1, y1]
            cls._old_line_coords = coords_list
            cls._first_dot_coords = None
            UISingleton.ui_main.unbind('<ButtonPress-1>')
        else:
            x, y = e.x, e.y
            coords_list = [x, y, x, y]
            cls._first_dot_coords = x, y

        line = UISingleton.canvas.create_line(*coords_list,
                                              width=cls._width,
                                              fill=cls._color)
        UISingleton.current_elements.append(line)

    @classmethod
    def change_params(cls):

        popup_menu = Menu(UISingleton.ui_main, tearoff=0)
        popup_menu.add_command(label='Change width', command=cls.change_width_func)
        cascade_menu = Menu(UISingleton.ui_main, tearoff=0)

        cascade_menu.add_command(label='white', command=lambda: cls._change_color_func('white'))

        cascade_menu.add_command(label='black', command=lambda: cls._change_color_func('black'))

        cascade_menu.add_command(label='red', command=lambda: cls._change_color_func('red'))

        cascade_menu.add_command(label='green', command=lambda: cls._change_color_func('green'))

        cascade_menu.add_command(label='cyan', command=lambda: cls._change_color_func('cyan'))

        cascade_menu.add_command(label='yellow', command=lambda: cls._change_color_func('yellow'))

        cascade_menu.add_command(label='magenta', command=lambda: cls._change_color_func('magenta'))

        popup_menu.add_cascade(label='Change Color', menu=cascade_menu)
        return popup_menu

    @classmethod
    def change_width_func(cls) -> None:
        new_w = simpledialog.askfloat(title='Write new width',
                                      prompt=f'Current is {cls._width}',
                                      parent=UISingleton.ui_main)
        cls._width = new_w
        cls._draw_line()

    @classmethod
    def _change_color_func(cls, color: str):
        cls._color = color
        cls._draw_line()