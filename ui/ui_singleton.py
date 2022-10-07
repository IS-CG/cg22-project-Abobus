import yaml
from dependency_injector import providers, containers

from ui._image_viewer import _ImageViewer


class UISingleton(containers.DeclarativeContainer):
    with open('.ui_config.yaml', 'r') as f:
        ui_conf = yaml.safe_load(f)
    ui_data = providers.Singleton(_ImageViewer)
