from concurrent import futures
from ray_tracer.engine.light import *
import time


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


class Tracer:
    def __init__(self, world: Geometry, light_sources: List['LightSource'], screen: 'Screen'):
        self.world = world
        self.lights = light_sources
        self.screen = screen

    @timing
    def run_parallel(self, iterations=2):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y, 3))

        iteration = 0
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            pixel_futures = [[
                self.submit_async(self.screen.screen[i][j], executor, iterations)
                for j in range(self.screen.res_y)] for i in range(self.screen.res_x)]

            for i in range(self.screen.res_x):
                for j in range(self.screen.res_y):
                    iteration += 1
                    if iteration % 20 == 1:
                        print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                    pixels[i, j, :] = pixel_futures[i][j].result().vec

        return pixels

    def submit_async(self, ray: Ray, executor, iterations):
        return executor.submit(Tracer.reflect_and_measure, ray, self.world, self.lights, iterations)

    def run(self, iterations=2):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y, 3))

        iteration = 0
        for i in range(self.screen.res_x):
            for j in range(self.screen.res_y):
                iteration += 1
                if iteration % 20 == 1:
                    print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                pixels[i, j, :] = self.reflect_and_measure(self.screen.screen[i][j], self.world, self.lights,
                                                        iterations).vec

        return pixels

    @staticmethod
    def reflect_and_measure(ray: Ray, world: Geometry, lights: List['LightSource'], iterations: int) -> Vector:

        if iterations == 0:
            return Vector([0, 0, 0])
        poly = ray.first_intersect(world)
        if poly is None:
            return Vector([0, 0, 0])

        strength = Tracer._strength_formula(ray, lights, poly, world)
        if poly.surface.k_reflect > 0:
            ray_reflect = ray.reflect(poly)
            strength += Tracer.reflect_and_measure(ray_reflect, world, lights, iterations-1)*poly.surface.k_reflect
        if poly.surface.k_refraction > 0:
            ray_refract = ray.refract(poly)
            strength += Tracer.reflect_and_measure(ray_refract, world, lights, iterations-1)*poly.surface.k_refraction

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

        return poly.surface.color*sum_strength


class Screen:
    def __init__(self, res_x: int, res_y: int, fov: int, scale=1):
        self.res_x = res_x
        self.res_y = res_y
        self.fov = fov

        self.screen = [[self._build_ray(i, j, scale) for j in range(res_y)] for i in range(res_x)]

    def _build_ray(self, i, j, scale):
        # put rays on a 2d grid (screen)
        offset = Vector([self._offset(i, self.res_x, scale), self._offset(j, self.res_y, scale), 0])
        # rotate all rays outwards to generate fov.
        direction = self._direction(self.fov, i, j)
        ray = Ray(offset=offset, direction=direction)
        return ray

    def _direction(self, fov, i, j):
        x_angle = self._angle(fov, i, self.res_x)
        y_angle = self._angle(fov, j, self.res_y)
        direction = Vector([0, 0, 1]) \
            .rotate(x_angle, Ray(Vector([0, 0, 0]), Vector([0, 1, 0]))) \
            .rotate(y_angle, Ray(Vector([0, 0, 0]), Vector([1, 0, 0])))
        return direction

    @staticmethod
    def _offset(i, resolution, scale):
        return - int(resolution * scale / 2) + i * scale

    @staticmethod
    def _angle(fov, i, res_x):
        return (fov / 2) * ((i - (res_x / 2)) / (res_x / 2))