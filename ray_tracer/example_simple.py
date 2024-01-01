import math
from pathlib import Path

from ray_tracer.engine.core import *

from ray_tracer.engine.utils import stl_to_polys
from PIL import Image
import random


if __name__ == "__main__":
    room_surface = Surface(k_ambient=50, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector.from_list([1, 1, 1]))
    object_surface = Surface(k_ambient=0, k_diffuse=0.2, k_reflect=0.8, k_refraction=0, refraction_strength=1.2, color=Vector.from_list([0.3, 0.7, 1]))
    elefant_surface = Surface(k_ambient=0, k_diffuse=1, k_reflect=0, k_refraction=0, refraction_strength=1.2,
                             color=Vector.from_list([0.7, 1, 0.3]))
    cube_surface = Surface(k_ambient=50, k_diffuse=1, k_reflect=0, k_refraction=0,
                             color=Vector.from_list([1, 0.5, 0.7]))


    width = 1000
    base = Vector.from_list([width - 400, 0, 500]) - Vector.from_list([width/2, width/2, width/2])
    floor = Square(
            base,
            base + Vector.from_list([0, width, 0]),
            base + Vector.from_list([0, 0, width]),
            base + Vector.from_list([0, width, width]), room_surface)

    sphere = Sphere(Vector.from_list([-50, 0, 400]), radius=200, surface=object_surface)

    cube = Cube.from_center(Vector.from_list([75, 0, 400]), width=50, surface=cube_surface)

    elefant = stl_to_polys(Vector.from_list([100, 0, 200]), Path(__file__).parent / "resources" / "Elephant.stl",
                           surface=elefant_surface)

    light_sources = [
        SphericalLightSource(100, Vector.from_list([-300, -100, 50]), 100, surface=None),
        SphericalLightSource(200, Vector.from_list([-100, 200, -150]), 50, surface=None),
        # LightSource(Vector.from_list([-300, 200, 0]), 50),
    ]

    world = ((floor + elefant + Geometry(sphere))
             .rotate(-5, Ray(offset=Vector.from_list([50, 0, 300]), direction=Vector.from_list([1, 0, 0])))
             .rotate(-20, Ray(offset=Vector.from_list([50, 0, 300]), direction=Vector.from_list([0, 1, 0])))
             .transpose(Vector.from_list([-100, 0, -150])))

    pixel_scale = 9
    screen = Screen(pixel_scale*60, pixel_scale*100, fov=75, scale=1/(pixel_scale*4))
    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)
    pixels = tracer.run_parallel_batched(iterations=5, light_iterations=10)

    # Plot
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data)
    image.show()
    image.save("generated_images/" + "simple_sphere" + str(random.randint(1, 10000)) + ".jpg", "JPEG")