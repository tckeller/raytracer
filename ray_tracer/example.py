from ray_tracer.engine.core import *
from PIL import Image
import random


if __name__ == "__main__":
    room_surface = Surface(k_ambient=50, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector([1, 1, 1]))
    room = Cube.from_center(Vector([0, 0, 599]), 1200, surface=room_surface)

    floor = Square(Vector([175, -100000, 0]), Vector([175, 100000, 0]), Vector([175, 100000, 200000]), Vector([175, -100000, 200000]), room_surface)

    # aq_surface = Surface(k_ambient=0, k_diffuse=0.1, k_reflect=0.3, k_refraction=0.6, refraction_streangth=1.5, color=Vector([0, 0, 1]))
    # aquarium = Cube.from_center(Vector([0, 0, 599]), 500, surface=aq_surface)

    sphere = Sphere(Vector([-300, 250, 800]), radius=250, surface=Surface(k_ambient=1, k_diffuse=0.2, k_reflect=0.8, k_refraction=0, color=Vector([0.9, 0.9, 0.9])))
    sphere2 = Sphere(Vector([50, -100, 300]), radius=100,
                    surface=Surface(k_ambient=1, k_diffuse=0.2, k_reflect=0, k_refraction=0.8,
                                    color=Vector([0, 1, 1])))

    cube_center = Vector([0, 0, 500])
    rotated_small_cube = Cube\
        .from_center(cube_center, 200,
                     surface=Surface(k_ambient=1, k_diffuse=0.1, k_reflect=0.2, k_refraction=0.7, color=Vector([0.7, 0.7, 0.3]))) \
        .rotate(45, Ray(cube_center, Vector([1, 0, 0]))) \
        .rotate(45, Ray(cube_center, Vector([0, 1, 0])))

    world = floor + rotated_small_cube + Geometry(sphere, sphere2)

    pixel_scale = 8
    screen = Screen(pixel_scale*60, pixel_scale*100, fov=75, scale=1/(pixel_scale*4))

    light_sources = [
        LightSource(Vector([-500, 0, -500]), 500),
    ]

    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

    pixels = tracer.run_parallel_batched(iterations=7)

    # Plot
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data)
    image.show()
    image.save("generated_images/" + "prism_" + str(random.randint(1, 10000)) + ".jpg", "JPEG")