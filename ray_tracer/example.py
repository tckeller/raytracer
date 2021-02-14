from ray_tracer.engine.light import *
from PIL import Image

if __name__ == "__main__":
    room = Cube.from_center(Vector([0, 0, 0]), 500)
    poly = Polygon(Vector([-50, 0, 150]), Vector([50, -50, 140]), Vector([50, 50, 160]))
    rotated_small_cube = Cube.from_center(Vector([0, 0, 100]), 10)\
        .rotate(45, Ray(Vector([0, 0, 100]), Vector([0, 1, 0])))\
        .rotate(45, Ray(Vector([0, 0, 100]), Vector([1, 0, 0])))
    world = room #+ rotated_small_cube

    screen = Screen(200, 200, fov=90)

    light_sources = [LightSource(Vector([50, 50, -100]), 1), LightSource(Vector([-100, -100, 100]), 1)]

    tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

    pixels = tracer.run_parallel(iterations=10, decay=0.3, max_diff=45)

    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = np.zeros((screen.res_x, screen.res_y, 3), dtype=np.uint8)

    data[:, :, 0] = pixels * 255
    data[:, :, 1] = pixels * 255
    data[:, :, 2] = pixels * 255

    image = Image.fromarray(data)
    print(pixels)
    image.show()