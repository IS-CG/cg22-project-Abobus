from lib.ui.ui_buider import UIBuilder

from lib.singleton_objects import UISingleton

if __name__ == "__main__":
    UIBuilder.build_ui()

    UISingleton.ui_main.mainloop()
