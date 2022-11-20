from dependency_injector import providers, containers
import tkinter as tk


class UISingleton(containers.DeclarativeContainer):
    ui_main = providers.Singleton(tk.Tk)
    canvas = providers.Singleton(None)
    main_menu = providers.Singleton(None)
    img_box = providers.Singleton(None)
