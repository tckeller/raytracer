from ray_tracer.engine.light import *
from ray_tracer.engine.core import *
from PIL import Image

if __name__ == "__main__":

    surface = Surface(k_ambient=10, k_diffuse=0.1, k_reflect=0.9)

    room = Cube.from_center(Vector([0, 0, 599]), 1200, surface=surface)

    rotated_small_cube = Cube.from_center(Vector([0, 0, 599]), 150, surface=Surface(k_ambient=10, k_diffuse=1, k_reflect=0)) \
        .rotate(45, Ray(Vector([0, 0, 599]), Vector([1, 0, 0]))) \
        .rotate(45, Ray(Vector([0, 0, 599]), Vector([0, 1, 0])))

    mirror = Square(Vector([-300, -300, 400]), Vector([300, -300, 400]), Vector([-300, 300, 400]), Vector([300, 300, 400]), surface=Surface(k_diffuse=0.1, k_reflect=0.9))

    world = room + rotated_small_cube

    screen = Screen(200, 200, fov=90, scale=1)

    light_sources = [LightSource(Vector([500, 0, 599]), 300),
                    LightSource(Vector([-500, -50, 599]), 50),
                    # LightSource(Vector([500, 0, -200]), 50),
                    LightSource(Vector([500, 500, 500]), 300),
                    ]

    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

    pixels = tracer.run_parallel(iterations=5, decay=0.3, max_diff=90)

    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = np.zeros((screen.res_x, screen.res_y, 3), dtype=np.uint8)

    data[:, :, 0] = pixels
    data[:, :, 1] = pixels
    data[:, :, 2] = pixels

    image = Image.fromarray(data)
    print(pixels)
    image.show()