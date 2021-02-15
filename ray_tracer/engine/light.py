from ray_tracer.engine.geometry import *


class LightSource:
    def __init__(self, pos: 'Vector', strength: float = 1):
        self.pos = pos
        self.strength = strength
