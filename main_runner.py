import yaml

from ui import ImageViewer

with open('configs/ui_config.yaml', 'r') as f:
    ui_conf = yaml.safe_load(f)

if __name__ == "__main__":
    ui = ImageViewer(ui_conf)
    ui.run_ui()
