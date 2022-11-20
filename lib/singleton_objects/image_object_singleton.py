from dependency_injector import providers, containers


class ImageObjectSingleton(containers.DeclarativeContainer):
    img = providers.Singleton(None)
    img_array = providers.Singleton(None)
    color = "RGB"
    gamma = 1.0
    default_img = providers.Singleton(None)
