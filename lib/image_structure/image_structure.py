from dependency_injector import providers, containers
from dataclasses import dataclass
from PIL import Image


@dataclass
class _ImageStructure:
    image: Image


class ImageStructure(containers.DeclarativeContainer):
    structure_data = providers.Singleton(_ImageStructure)
