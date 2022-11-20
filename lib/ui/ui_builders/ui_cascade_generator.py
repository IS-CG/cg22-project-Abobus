from typing import Callable, List
import types
import functools
import cv2
from tkinter import Menu

from lib.singleton_objects import UISingleton
from copy import deepcopy


def new_func(f: Callable, name: str) -> Callable:
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=name,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


class UICascadeGenerator:
    @staticmethod
    def generate_cascade(cascade_pattern_func: Callable,
                         cascade_items: List[str]) -> Menu:

        cascade_menu = Menu(UISingleton.ui_main, tearoff=0)

        for item in cascade_items:
            cascade_menu.add_command(label=item,
                                     command=new_func(lambda: cascade_pattern_func(item),
                                                      f'new_{item}_func'))

        return cascade_menu

