from typing import List
from concurrent import futures
from copy import copy, deepcopy
from ray_tracer.engine.light import *


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
                            Tracer.reflect_and_measure, self.screen.screen[i][j], self.world, self.lights,iterations)
                    )

            for i in range(self.screen.res_x):
                for j in range(self.screen.res_y):
                    iteration += 1
                    if iteration % 20 == 1:
                        print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                    pixels[i, j] = min(pixel_futures[i][j].result(), 255)

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
                                                        iterations)

        return pixels

    @staticmethod
    def reflect_and_measure(ray: Ray, world: Geometry, lights: List['LightSource'], iterations: int) -> float:

        if iterations == 0:
            return 0
        ray, poly = ray.multi_reflect(world)
        if ray is None:
            return 0

        strength = Tracer._strength_formula(ray, lights, poly, world)
        if poly.surface.k_reflect > 0:
            return Tracer.reflect_and_measure(ray, world, lights, iterations-1) \
                * poly.surface.k_reflect \
                + strength
        else:
            return strength

    @staticmethod
    def _strength_formula(ray: Ray, lights: List['LightSource'], poly: Polygon, world: Geometry):
        sum_strength = poly.surface.k_ambient

        for light in lights:
            try:
                shadows = Ray(offset=ray.offset, direction=light.pos - ray.offset).distances_to_polys(world)
                if np.nanmin(shadows) < 1:
                    # shadow thrower closer than light
                    continue
            except ValueError:
                pass
            strength = light.strength * \
                poly.surface.k_diffuse * (abs(poly.normal_vector.dot((ray.offset - light.pos).norm())))

            sum_strength += strength

        return sum_strength


class Screen:
    def __init__(self, res_x: int, res_y: int, fov: int, scale=1):
        screen = []
        self.res_x = res_x
        self.res_y = res_y
        self.fov = fov
        for i in range(res_x):
            screen.append([])
            for j in range(res_y):
                offset = Vector([- int(res_x*scale/2) + i*scale, - int(res_y*scale/2) + j*scale, 0])

                x_angle = (fov/2) * ((i-(res_x/2))/(res_x/2))
                y_angle = (fov/2) * ((j-(res_y/2))/(res_y/2))
                direction = Vector([0, 0, 1])\
                    .rotate(x_angle, Ray(Vector([0, 0, 0]), Vector([0, 1, 0])))\
                    .rotate(y_angle, Ray(Vector([0, 0, 0]), Vector([1, 0, 0])))
                screen[i].append(Ray(offset=offset, direction=direction))
        self.screen = screen