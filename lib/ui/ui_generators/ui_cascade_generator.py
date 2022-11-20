from typing import Callable, List

from tkinter import Menu

from lib.singleton_objects import UISingleton
from copy import deepcopy


class UICascadeGenerator:
    @staticmethod
    def generate_cascade(cascade_pattern_func: Callable,
                         cascade_items: List[str]) -> Menu:

        cascade_menu = Menu(UISingleton.ui_main, tearoff=0)

        def generator():
            for item in cascade_items:

                def item_func() -> Callable:
                    return cascade_pattern_func(item)

                yield item_func

        for func_label, func in zip(cascade_items, generator()):
            cascade_menu.add_command(label=func_label, command=func)

        return cascade_menu
