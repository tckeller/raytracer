from ray_tracer.engine.core import *
from PIL import Image
import random


if __name__ == "__main__":
    room_surface = Surface(k_ambient=50, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector([1, 1, 1]))
    room = Cube.from_center(Vector([0, 0, 599]), 1200, surface=room_surface)

    sphere = Sphere(Vector([50, -100, 300]), radius=100,
                    surface=Surface(k_ambient=1, k_diffuse=1, color=Vector([0, 1, 1])))

    world = room + Geometry(sphere)

    pixel_scale = 4
    screen = Screen(pixel_scale*60, pixel_scale*100, fov=75, scale=1/(pixel_scale*4))

    light_sources = [
        LightSource(Vector([-500, 0, -500]), 500),
    ]

    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

    pixels = tracer.run(iterations=1)

    # Plot
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data)
    image.show()
    image.save("generated_images/" + "simple_" + str(random.randint(1, 10000)) + ".jpg", "JPEG")