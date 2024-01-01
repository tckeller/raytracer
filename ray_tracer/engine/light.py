from ray_tracer.engine.geometry import *
from numpy.random import uniform


class LightSource:
    def __init__(self, pos: 'Vector', strength: float = 1):
        self.pos = pos
        self.strength = strength

    def random_pos_on_surface(self):
        return self.pos


class SphericalLightSource(Sphere):
    def __init__(self, strength: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength = strength

    def random_pos_on_surface(self):
        random_point_in_sphere = uniform(-self.radius, self.radius, 3)
        return self.pos + Vector(random_point_in_sphere)
