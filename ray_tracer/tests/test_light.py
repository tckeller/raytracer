from unittest import TestCase
from ray_tracer.engine.geometry import *
from ray_tracer.engine.light import *
from PIL import Image


class TestTracer(TestCase):

    def test_runs(self):
        room = Cube.from_center(Vector([0, 0, 0]), 500)
        poly = Polygon(Vector([-50, 0, 150]), Vector([50, -50, 140]), Vector([50, 50, 160]))
        #rotated_small_cube = Cube.from_center(Vector([0, 0, 100]), 10)\
        #    .rotate(45, Ray(Vector([0, 0, 100]), Vector([0, 1, 0])))\
        #    .rotate(45, Ray(Vector([0, 0, 100]), Vector([1, 0, 0])))
        world = room #+ Geometry(poly)

        screen = Screen(100, 100, fov=90)

        light_sources = [LightSource(Vector([50, 50, -100]), 0.3), LightSource(Vector([-100, -100, 100]), 0.2)]

        tracer = Tracer(world=world, light_sources=light_sources, screen=screen)

        pixels = tracer.run(iterations=5, decay=0.75, max_diff=90)

        # Create a 1024x1024x3 array of 8 bit unsigned integers
        data = np.zeros((screen.res_x, screen.res_y, 3), dtype=np.uint8)

        data[:, :, 0] = pixels * 255
        data[:, :, 1] = pixels * 255
        data[:, :, 2] = pixels * 255

        image = Image.fromarray(data)
        print(pixels)
        image.show()

    def test_specific_pixel(self):
        room = Cube.from_center(Vector([0, 0, 0]), 500)
        rotated_small_cube = Cube.from_center(Vector([0, 0, 100]), 50)\
            .rotate(45, Ray(Vector([0, 0, 100]), Vector([0, 1, 0])))\
            .rotate(45, Ray(Vector([0, 0, 100]), Vector([1, 0, 0])))
        world = room + rotated_small_cube

        screen = Screen(30, 30, 90)
        screen.screen = [[Ray(offset=Vector([-14, -4, 0]),
                              direction=Vector([-0.6691306063588582, 0.1923400341029386, 0.7178227796016974]))]]

        tracer = Tracer(world=world, light_sources=[Vector([0, 0, 200])], screen=screen)

        pixels = tracer.run(iterations=5)
        print(pixels)
