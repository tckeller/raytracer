import warnings
from concurrent import futures
from ray_tracer.engine.light import *
import time
from tqdm import tqdm
import math

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
    def run_parallel(self, iterations=2, light_iterations=1):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y, 3))

        iteration = 0
        with futures.ProcessPoolExecutor(max_workers=32) as executor:
            pixel_futures = [[
                self.submit_async(self.screen.screen[i][j], executor, iterations, light_iterations)
                for j in range(self.screen.res_y)] for i in range(self.screen.res_x)]

            for i in range(self.screen.res_x):
                for j in range(self.screen.res_y):
                    iteration += 1
                    if iteration % 20 == 1:
                        print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                    pixels[i, j, :] = pixel_futures[i][j].result().vec

        return pixels

    def process_ray_batch(self, rays_batch, iterations, light_iterations):
        # This function processes a list of rays corresponding to a batch of pixels
        batch_results = []
        for ray in rays_batch:
            color = Tracer.reflect_and_measure(ray, self.world, self.lights,
                                               iterations, 1, light_iterations)  # This is your actual ray processing
            batch_results.append(color)  # Assuming color is a Vector; adjust as necessary

        return batch_results

    @timing
    def run_parallel_batched(self, iterations=2, light_iterations=1, batch_size=None):
        if batch_size is None:
            batch_size = int(self.screen.res_y / math.sqrt(30))

        pixels = np.zeros((self.screen.res_x, self.screen.res_y, 3))

        # Calculate the total number of batches
        total_batches = ((self.screen.res_x + batch_size - 1) // batch_size) * (
                    (self.screen.res_y + batch_size - 1) // batch_size)

        with futures.ProcessPoolExecutor() as executor:
            all_rays = [[self.screen.screen[i][j] for j in range(self.screen.res_y)] for i in range(self.screen.res_x)]

            future_to_pixel = {}

            # Here, we create a tqdm object that will track the progress
            with tqdm(total=total_batches, desc="Processing batches", unit="batch") as progress_bar:
                for start_x in range(0, self.screen.res_x, batch_size):
                    for start_y in range(0, self.screen.res_y, batch_size):
                        rays_batch = [all_rays[i][j] for i in
                                      range(start_x, min(start_x + batch_size, self.screen.res_x))
                                      for j in range(start_y, min(start_y + batch_size, self.screen.res_y))]

                        future = executor.submit(self.process_ray_batch, rays_batch, iterations, light_iterations)
                        future_to_pixel[future] = (start_x, start_y)

                # As tasks complete, we update the progress bar.
                for future in futures.as_completed(future_to_pixel):
                    start_x, start_y = future_to_pixel[future]
                    batch_results = future.result()
                    idx = 0

                    for i in range(start_x, min(start_x + batch_size, self.screen.res_x)):
                        for j in range(start_y, min(start_y + batch_size, self.screen.res_y)):
                            pixels[i, j, :] = batch_results[idx]
                            idx += 1

                    # Update the progress bar
                    progress_bar.update(1)

        return pixels

    def submit_async(self, ray: Ray, executor, iterations, light_iterations):
        return executor.submit(Tracer.reflect_and_measure, ray, self.world, self.lights, iterations, 1, light_iterations)

    def run(self, iterations=2, light_iterations=1):
        pixels = np.zeros((self.screen.res_x, self.screen.res_y, 3))

        iteration = 0
        for i in range(self.screen.res_x):
            for j in range(self.screen.res_y):
                iteration += 1
                if iteration % 20 == 1:
                    print(f"{iteration*100/(self.screen.res_x*self.screen.res_y)} %")
                pixels[i, j, :] = self.reflect_and_measure(self.screen.screen[i][j], self.world, self.lights,
                                                        iterations, 1, light_iterations)

        return pixels

    @staticmethod
    def reflect_and_measure(ray: Ray, world: Geometry, lights: List['LightSource'], iterations: int, is_inside: bool = False, light_iterations=1) -> Vector:

        if ray is None:
            return Vector.from_list([0, 0, 0])
        if iterations == 0:
            return Vector.from_list([0, 0, 0])
        poly = ray.first_intersect(world)
        if poly is None:
            return Vector.from_list([0, 0, 0])
        else:
            intersect_point, intersec_dist = ray.intersect(poly)

        strength = Tracer._strength_formula(ray, intersect_point, lights, poly, world, light_iterations)
        total_reflect = False
        if poly.surface.k_refraction > 0:
            ray_refract, total_reflect = ray.refract(poly, intersect_point, is_inside)
            if ray_refract:
                strength += (Tracer.reflect_and_measure(ray_refract, world, lights, iterations-1, poly.surface.refraction_strength)*poly.surface.k_refraction).elementwise_mul(poly.surface.color(intersect_point))
        if poly.surface.k_reflect > 0 or total_reflect:
            ray_reflect = ray.reflect(poly, intersect_point)
            reflect_coeff = 1 if total_reflect else poly.surface.k_reflect
            strength += (Tracer.reflect_and_measure(ray_reflect, world, lights, iterations-1, not is_inside)*reflect_coeff).elementwise_mul(poly.surface.color(intersect_point))

        strength = Vector([min(val, 255) for val in strength])
        return strength

    @staticmethod
    def _strength_formula(ray: Ray, intersect_point: Vector, lights: List['LightSource'], poly: Polygon, world: Geometry, light_iterations: int = 1):
        sum_strength = poly.surface.k_ambient

        for light in lights:
            min_light_iterations = 10  # minimum number of iterations to get a baseline
            tolerance = 4  # variance tolerance level to determine if more samples are needed

            sum_light_iterations = 0.0

            # Variables for Welford's method
            mean = 0.0
            M2 = 0.0
            variance = 0.0

            for light_iter in range(1, light_iterations + 1):  # light_iter should start from 1 for this algorithm

                light_pos = light.random_pos_on_surface()

                normal_vector = poly.normal_vector(intersect_point)
                if (-ray.direction).dot(normal_vector) < 0:
                    normal_vector = -normal_vector

                cos_angle = (normal_vector.dot((light_pos - intersect_point).norm()))
                if cos_angle < 0:
                    continue

                shadows = Ray(offset=intersect_point, direction=light_pos - intersect_point).distances_to_polys(world)

                distance_to_light = intersect_point.distance(light_pos)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if np.nanmin(shadows) < distance_to_light:
                        # shadow thrower closer than light
                        continue

                # Calculate light contribution
                light_contribution = light.strength * poly.surface.k_diffuse * cos_angle
                sum_light_iterations += light_contribution

                # Welford's online variance calculation
                delta = light_contribution - mean
                mean += delta / light_iter
                delta2 = light_contribution - mean
                M2 += delta * delta2  # This is the sum of squared differences from the current mean

                # Ensure we have enough samples before making decisions based on variance
                if light_iter >= min_light_iterations:
                    variance = M2 / (light_iter - 1)  # Use a sample variance formula, with "n-1" in the denominator

                    if variance < tolerance:
                        break  # If the variance is below the threshold, we assume the estimate is good enough

            # Final average based on the actual number of iterations
            sum_strength += sum_light_iterations / light_iter

        return poly.surface.color(intersect_point)*sum_strength


class Screen:
    def __init__(self, res_x: int, res_y: int, fov: int, scale=1):
        self.res_x = res_x
        self.res_y = res_y
        self.fov = fov

        self.screen = [[self._build_ray(i, j, scale) for j in range(res_y)] for i in range(res_x)]

    def _build_ray(self, i, j, scale):
        # put rays on a 2d grid (screen)
        offset = Vector.from_list([self._offset(i, self.res_x, scale), -self._offset(j, self.res_y, scale), 0])
        # rotate all rays outwards to generate fov.
        direction = self._direction(self.fov, i, j)
        ray = Ray(offset=offset, direction=direction)
        return ray

    def _direction(self, fov, i, j):
        x_angle = self._angle(fov, i, self.res_x)
        y_angle = self._angle(fov, j, self.res_y)
        direction = Vector.from_list([0, 0, 1]) \
            .rotate(x_angle, Ray(Vector.from_list([0, 0, 0]), Vector.from_list([0, 1, 0]))) \
            .rotate(y_angle, Ray(Vector.from_list([0, 0, 0]), Vector.from_list([1, 0, 0])))
        return direction

    @staticmethod
    def _offset(i, resolution, scale):
        return - int(resolution * scale / 2) + i * scale

    @staticmethod
    def _angle(fov, i, res_x):
        return - fov/2 + i*fov/res_x

