from dependency_injector import providers, containers
import tkinter as tk


class UISingleton(containers.DeclarativeContainer):
    ui_main = providers.Singleton(tk.Tk)
    panel = providers.Singleton(None)
    main_menu = providers.Singleton(None)
