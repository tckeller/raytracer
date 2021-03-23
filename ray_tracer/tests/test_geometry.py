from unittest import TestCase
from ray_tracer.engine.geometry import *


class TestVector(TestCase):
    def test_addition(self):
        a = Vector([1, 2, 3])
        b = Vector([2, 3, 4])
        self.assertEqual(a + b, Vector([3, 5, 7]))

    def test_subtraction(self):
        a = Vector([1, 2, 3])
        b = Vector([2, 3, 4])
        self.assertEqual(a - b, Vector([-1, -1, -1]))

    def test_div(self):
        a = Vector([2, 4, 6])
        self.assertEqual(a / 2, Vector([1, 2, 3]))

    def test_mul(self):
        a = Vector([1, 2, 3])
        self.assertEqual(a * 3, Vector([3, 6, 9]))

    def test_dot_product(self):
        a = Vector([1, 2, 3])
        b = Vector([2, 3, 4])
        self.assertEqual(a.dot(b), 2 + 6 + 12)

    def test_cross(self):
        a = Vector([1, 2, 3])
        b = Vector([2, 3, 4])
        self.assertEqual(a.cross(b), Vector([8 - 9, 6 - 4, 3 - 4]))

    def test_rotate(self):
        a = Vector([1, 0, 0])
        rotation_axis = Ray(offset=Vector([0, 0, 0]), direction=Vector([0, 0, 1]))
        self.assertEqual(a.rotate(180, rotation_axis).round(5), Vector([-1, 0, 0]))

    def test_rotate_offset(self):
        a = Vector([1, 1, 1])
        rotation_axis = Ray(offset=Vector([0, 1, 1]), direction=Vector([0, 0, 1]))
        self.assertEqual(a.rotate(180, rotation_axis).round(5), Vector([-1, 1, 1]))

    def test_angle(self):
        a = Vector([1, 0, 0])
        b = Vector([-1, 0, 0])
        self.assertEqual(a.angle(b), 180)


class TestRay(TestCase):
    def test_intersect(self):
        a = Ray(Vector([0, 0, 0]), Vector([0, 0, 1]))
        poly = Polygon(Vector([1, 0, 2]), Vector([-1, -1, 2]), Vector([-1, 1, 2]), Surface(1, 1, 0))
        a.intersect(poly)

        intersect, distance = a.intersect(poly)
        self.assertEqual(intersect, Vector([0, 0, 2]))
        self.assertEqual(distance, 2)

    def test_reflect(self):
        a = Ray(Vector([1, 0, 0]), Vector([-1, 1, 0]))
        poly = Polygon(Vector([1, 1, 1]), Vector([0, 1, -1]), Vector([-1, 1, -1]), Surface(1, 1, 0))
        reflection = a.reflect(poly)
        self.assertEqual(reflection.offset, Vector([0, 1, 0]))
        self.assertEqual(reflection.direction, Vector([-1, -1, 0]))

    def test_multi_reflect(self):
        a = Ray(Vector([1, 0, 0]), Vector([-1, 1, 0]))
        poly1 = Polygon(Vector([1, 1, 1]), Vector([0, 1, -1]), Vector([-1, 1, -1]), Surface(1, 1, 0))
        poly2 = Polygon(Vector([1, 2, 1]), Vector([0, 2, -1]), Vector([-1, 2, -1]), Surface(1, 1, 0))
        poly3 = Polygon(Vector([1, -1, 1]), Vector([0, -1, -1]), Vector([-1, -1, -1]), Surface(1, 1, 0))
        world = Geometry(*[poly1, poly2, poly3])

        reflection, _ = a.multi_reflect(world)
        self.assertEqual(reflection.offset, Vector([0, 1, 0]))
        self.assertEqual(reflection.direction, Vector([-1, -1, 0]))


class TestPolygon(TestCase):
    def test_is_inside(self):
        poly = Polygon(Vector([1, 0, 2]), Vector([-1, -1, 2]), Vector([-1, 1, 2]), Surface(1, 1, 0))
        self.assertTrue(poly.is_inside(Vector([1, 0, 2])))

    def test_is_not_inside(self):
        poly = Polygon(Vector([1, 0, 2]), Vector([-1, -1, 2]), Vector([-1, 1, 2]), Surface(1, 1, 0))
        self.assertFalse(poly.is_inside(Vector([2, 0, 2])))

    def test_rotate(self):
        poly = Polygon(Vector([1, 0, 2]), Vector([-1, -1, 2]), Vector([-1, 1, 2]), Surface(1, 1, 0))
        rotation_axis = Ray(offset=Vector([0, 0, 2]), direction=Vector([0, 0, 1]))
        print(poly.rotate(180, rotation_axis))


class TestGeometry(TestCase):
    def test_rotate(self):
        square = Square(Vector([10, 10, 0]), Vector([10, 12, 0]), Vector([12, 10, 0]), Vector([12, 12, 0]), Surface(1, 1, 0))
        rotation_axis = Ray(offset=Vector([11, 11, 0]), direction=Vector([0, 0, 1]))
        self.assertEqual(square, square.rotate(90, rotation_axis))

    def test_rotate_cube(self):
        cube = Cube.from_center(Vector([0, 0, 100]), 100, Surface(1, 1, 0))
        rotation_axis1 = Ray(offset=Vector([0, 0, 100]), direction=Vector([0, 1, 0]))
        rotation_axis2 = Ray(offset=Vector([0, 0, 100]), direction=Vector([1, 0, 0]))
        #self.assertEqual(cube.rotate(180, rotation_axis1).rotate(180, rotation_axis2), cube)
