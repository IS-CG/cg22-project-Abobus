from abc import ABC

class IPainter:
    @classmethod
    def draw(cls):
        raise NotImplemented

    @classmethod
    def _draw(cls, e):
        raise NotImplemented

    @classmethod
    def change_params(cls):
        raise NotImplemented
