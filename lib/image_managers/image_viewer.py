from functools import partial

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import cv2

from lib.singleton_objects import UISingleton, ImageObjectSingleton


class ImageViewer:

    @staticmethod
    def display_img():
        image = ImageTk.PhotoImage(ImageObjectSingleton.img)
        UISingleton.canvas.itemconfig(UISingleton.img_box, image=image)
        UISingleton.canvas.image = image

    @classmethod
    def auto_image_resizer(cls, e):
        if type(ImageObjectSingleton.img_array) != np.ndarray:
            return None

        img = ImageObjectSingleton.img_array
        h, w = ImageObjectSingleton.img_array.shape[:2]
        if e.width < w or e.height < h:
            new_w = e.width if e.width < w else w
            new_h = e.height if e.height < h else h
            resized_bg = cv2.resize(img, (new_w, new_h))
            cls.display_img_array(resized_bg)
    
    @staticmethod
    def stash_changes():
        ImageObjectSingleton.img_array = ImageObjectSingleton.default_img
        ImageObjectSingleton.color = "RGB"
        for element in UISingleton.current_elements:
            UISingleton.canvas.delete(element)
        ImageViewer.display_img_array(ImageObjectSingleton.default_img)
    

    @classmethod
    def preview_img(cls, data: np.ndarray) -> None:
        top = tk.Toplevel(UISingleton.ui_main)
        top.geometry("700x250")
        top.title("Child Window")
        child_menu = tk.Menu(top)
        top.config(menu=child_menu)
        child_menu.add_command(label='Save Current Img', command=partial(cls.display_img_array, data))
        canvas = tk.Canvas(top, highlightthickness=0, width=600, height=400)
        canvas.pack(fill=tk.BOTH, expand=1)
        img_box = canvas.create_image(0, 0, image=None, anchor='nw')
        img = ImageTk.PhotoImage(Image.fromarray(data))
        canvas.itemconfig(img_box, image=img)
        canvas.image = img

    @staticmethod
    def get_new_img_center_by_user_input():
        new_center_x = int(simpledialog.askstring("Input", "Enter center X coordinate:"))
        new_center_y = int(simpledialog.askstring("Input", "Enter center Y coordinate:"))

        x = int(new_center_x - ImageObjectSingleton.img.width // 2)
        y = int(new_center_y - ImageObjectSingleton.img.height // 2)

        return x, y


    @classmethod
    def move_img(cls):
        """
        Move image to new center by user input
        """
        choice = messagebox.askquestion(title='Change center coords',
                                        message='Click "No" to mouse transfer, Click "Yes" to enter coordinates',                                        parent=UISingleton.ui_main,
                                        type='yesnocancel', default='yes', icon='question')

        if choice == 'no':
            UISingleton.canvas.bind("<Button-1>", cls.move_by_mouse_click)
        elif choice == 'yes':
            x_coord, y_coord = cls.get_new_img_center_by_user_input()
            UISingleton.canvas.move(UISingleton.img_box, x_coord, y_coord)
        else:
            pass

    @staticmethod
    def move_by_mouse_click(event) -> tuple[int, int]:
        """
        return user click coordinates
        """
        UISingleton.canvas.coords(UISingleton.img_box, event.x, event.y)
        UISingleton.canvas.unbind("<Button-1>")


    @classmethod
    def display_img_array(cls, data: np.ndarray) -> None:
        """
        Display image from array
        data: np.ndarray - image array
        """
        ImageObjectSingleton.img = Image.fromarray(data)
        cls.display_img()
