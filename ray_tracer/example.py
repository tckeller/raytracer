import math
import numpy
from pathlib import Path

from ray_tracer.engine.core import *
from PIL import Image
from ray_tracer.engine.utils import stl_to_polys


def colormap(v: Vector) -> Vector:
    return np.sin(v * Vector.from_list([0, 1, 0]) / 25) + Vector.from_list([0,1,0])

if __name__ == "__main__":

    room_surface = Surface(k_ambient=1, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector.from_list([1, 1, 1]))
    reflective_surface = Surface(k_ambient=1, k_diffuse=0.5, k_reflect=0.5, k_refraction=0, color=Vector.from_list([0.9, 0.9, 0.9]))
    transparent_surface = Surface(k_ambient=1, k_diffuse=0.3, k_reflect=0, k_refraction=0.7, color=Vector.from_list([0.7, 0.9, 0.1]))
    green_surface = Surface(k_ambient=1, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector.from_list([0, 1, 1]))

    colormap_surface = Surface(k_ambient=1, k_diffuse=1, k_reflect=0, k_refraction=0, colormap=colormap)

    # Things I can put into the scene
    room = Cube.from_center(Vector.from_list([0, 0, 200]), 405, surface=room_surface)

    floor = Square(Vector.from_list([175, 100000, -100000]), Vector.from_list([175, -100000, -100000]), Vector.from_list([175, 100000, 100000]), Vector.from_list([175, -100000, 100000]), room_surface)

    grid_size = 6
    sphere_grid = [
        Sphere(Vector.from_list([-100, 100-(grid_size / 2) * 200 + i*200, 100 + j*200]),
               radius=50, surface=reflective_surface)
        for i in range(grid_size) for j in range(grid_size)]

    cube_center = Vector.from_list([0, 0, 500])
    rotated_small_cube = Cube\
        .from_center(cube_center, 200,
                     surface=transparent_surface) \
        .rotate(45, Ray(cube_center, Vector.from_list([1, 0, 0]))) \
        .rotate(45, Ray(cube_center, Vector.from_list([0, 1, 0])))

    colormap_sphere = Sphere(Vector.from_list([202, 0, 200]), radius=75, surface=colormap_surface)
    rabbit = stl_to_polys(Vector.from_list([0, 0, 200]), Path(__file__).parent / "resources" / "rabbit.stl",
                          surface=green_surface)

    # Lights
    light_sources = [
        SphericalLightSource(strength=200, pos=Vector.from_list([-100, 0, 300]), radius=10, surface=None),
        SphericalLightSource(strength=100, pos=Vector.from_list([100, 0, 50]), radius=10, surface=None),
        SphericalLightSource(strength=100, pos=Vector.from_list([-200, 0, 50]), radius=10, surface=None),
    ]

    # Set up world
    world = rabbit + room

    # Render
    pixel_scale = 2
    screen = Screen(pixel_scale*60, pixel_scale*100, fov=90, scale=1/(pixel_scale*4))
    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)
    pixels = tracer.run_parallel_batched(iterations=3, light_iterations=5)

    # Plot
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data)
    image.show()
    image.save("generated_images/" + "rabbit_" + str(time.time()) + ".jpg", "JPEG")