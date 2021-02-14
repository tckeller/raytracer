from ray_tracer.engine.geometry import *
from typing import List
from concurrent import futures
from copy import copy, deepcopy

class Tracer:
    def __init__(self, world: Geometry, light_sources: List['LightSource'], screen: 'Screen'):
        self.world = world
        self.lights = light_sources
        self.screen = screen

    def run_parallel(self, iterations=2, decay: float = 0.9, max_diff: float = 90):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y))

        iteration = 0
        pixel_futures = []
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            for i in range(self.screen.res_x):
                pixel_futures.append([])
                for j in range(self.screen.res_y):
                    pixel_futures[i].append(
                        executor.submit(
                            Tracer.reflect_and_measure,
                            deepcopy(self.screen.screen[i][j]),
                            deepcopy(self.world),
                            deepcopy(self.lights),
                            deepcopy(iterations), deepcopy(decay), max_diff))

            for i in range(self.screen.res_x):
                for j in range(self.screen.res_y):
                    iteration += 1
                    if iteration % 20 == 1:
                        print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                    pixels[i, j] = pixel_futures[i][j].result()

        return pixels

    def run(self, iterations=2, decay: float = 0.9, max_diff: float = 90):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y))

        iteration = 0
        for i in range(self.screen.res_x):
            for j in range(self.screen.res_y):
                iteration += 1
                if iteration % 20 == 1:
                    print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                pixels[i, j] = self.reflect_and_measure(self.screen.screen[i][j], self.world, self.lights,
                                                        iterations, decay, max_diff)

        return pixels

    @staticmethod
    def reflect_and_measure(ray: Ray, world: Geometry, lights: List['LightSource'], iterations: int, decay: float,
                            max_diff: float, recursion: int = 0, distances: List[float] = None) -> float:

        if distances is None:
            distances = ray.distances_to_polys(world)

        if iterations == 0:
            return 0
        ray = ray.multi_reflect(world, distances)
        if ray is None:
            return 0

        distances = ray.distances_to_polys(world)

        try:
            min_distance = np.nanmin(distances)
        except ValueError:
            min_distance = None

        strength = Tracer._strength_formula(ray, max_diff, min_distance, lights)
        return min(Tracer.reflect_and_measure(ray, world, lights, iterations-1, decay, max_diff, recursion+1, distances)
                   + strength*(np.power(decay, recursion)), 1)

    @staticmethod
    def _strength_formula(ray: Ray,  max_diff: float, min_distance: float, lights: List['LightSource']):
        sum_strength = 0

        for light in lights:
            if min_distance is None or (light.pos - ray.offset).abs() > min_distance:
                continue
            angle = ray.direction.angle(light.pos - ray.offset)
            # strength = (1 - min(abs(angle) / max_diff, 1)) * light.strength
            strength = light.strength/3 if angle < max_diff else 0

            sum_strength += strength

        return min(sum_strength, 1)


class Screen:
    def __init__(self, res_x: int, res_y: int, fov: int):
        screen = []
        self.res_x = res_x
        self.res_y = res_y
        self.fov = fov
        for i in range(res_x):
            screen.append([])
            for j in range(res_y):
                offset = Vector([- int(res_x/2) + i, - int(res_y/2) + j, 0])

                x_angle = (fov/2) * ((i-(res_x/2))/(res_x/2))
                y_angle = (fov/2) * ((j-(res_y/2))/(res_y/2))
                direction = Vector([0, 0, 1])\
                    .rotate(x_angle, Ray(Vector([0, 0, 0]), Vector([0, 1, 0])))\
                    .rotate(y_angle, Ray(Vector([0, 0, 0]), Vector([1, 0, 0])))
                screen[i].append(Ray(offset=offset, direction=direction))
        self.screen = screen


class LightSource:
    def __init__(self, pos: Vector, strength: float = 1):
        self.pos = pos
        self.strength = strength
