from ray_tracer.engine.core import *
from PIL import Image


if __name__ == "__main__":
    room_surface = Surface(k_ambient=10, k_diffuse=1, k_reflect=0, k_refraction=0, color=Vector([1, 1, 1]))
    room = Cube.from_center(Vector([0, 0, 599]), 1200, surface=room_surface)

    aq_surface = Surface(k_ambient=0, k_diffuse=0.1, k_reflect=0.3, k_refraction=0.6, refraction_streangth=1.5, color=Vector([0, 0, 1]))
    aquarium = Cube.from_center(Vector([0, 0, 599]), 500, surface=aq_surface)

    rotated_small_cube = Cube\
        .from_center(Vector([0, 0, 599]), 150,
                     surface=Surface(k_ambient=1, k_diffuse=0.5, k_reflect=0.5, k_refraction=0, color=Vector([1, 0, 1]))) \
        .rotate(45, Ray(Vector([0, 0, 599]), Vector([1, 0, 0]))) \
        .rotate(45, Ray(Vector([0, 0, 599]), Vector([0, 1, 0])))

    world = room + rotated_small_cube + aquarium

    screen = Screen(200, 300, fov=90, scale=1)

    light_sources = [
        LightSource(Vector([500, 0, 599]), 500),
        LightSource(Vector([-500, -50, 599]), 100),
        LightSource(Vector([50, -50, -100]), 1000),]

    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

    pixels = tracer.run_parallel(iterations=10)

    # Plot
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data)
    image.show()
