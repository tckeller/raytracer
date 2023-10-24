import numpy
import numpy as np
from typing import Union, List, Tuple, Optional
from scipy.spatial.transform import Rotation
from math import acos, cos, sin, sqrt
from ray_tracer.engine.intersect import all_distances

class Vector(np.ndarray):

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        return obj

    @staticmethod
    def from_list(vec: List[float]):
        return Vector(np.array(vec).astype(np.float64))

    def distance(self, other: 'Vector') -> 'Vector':
        return (self - other).abs()

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(np.cross(self, other))

    def abs(self):
        return np.linalg.norm(self)

    def norm(self) -> 'Vector':
        return self / self.abs()

    def rotate(self, angle: float, axis: 'Ray'):
        rotation_radians = np.radians(angle)
        rotation_axis = axis.direction.norm()
        rotation_vector = rotation_axis * rotation_radians
        rotation = Rotation.from_rotvec(rotation_vector)
        rotated_vec = Vector(rotation.apply((self - axis.offset)).astype(np.float64)) + axis.offset
        return rotated_vec

    def angle(self, other: 'Vector'):
        return (acos(self.dot(other) / (self.abs()*other.abs())))*(360/(2*np.pi))

    def elementwise_mul(self, other: 'Vector'):
        return numpy.multiply(self, other)

    def __str__(self) -> str:
        return f"Vector({self[0]}, {self[1]}, {self[2]})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return int(self.sum())

    def __eq__(self, other: 'Vector') -> bool:
        return np.array_equal(self, other)


class Ray:
    def __init__(self, offset: Vector, direction: Vector):
        self.offset = offset
        self.direction = direction.norm()

    def point_on_line(self, scaler: float) -> Tuple['Vector', float]:
        return self.offset + self.direction * scaler, scaler

    def offset_distance(self, point: Vector):
        return self.offset.distance(point)

    def intersect(self, poly: 'Polygon') -> Tuple[Optional[Vector], Optional[float]]:
        if isinstance(poly, Polygon):
            intersect, distance = self.point_on_line(
                (poly.a - self.offset).dot(poly.normal_vec) / (self.direction.dot(poly.normal_vec))
            )
            return intersect, distance
        if isinstance(poly, Sphere):
            intersect_vector = poly.closest_intersect_distance(self)
            return (None, None) if intersect_vector is None else (intersect_vector, self.offset_distance(intersect_vector))

    def first_intersect(self, world: 'Geometry') -> Union[None, 'Polygon']:
        distances = self.distances_to_polys(world)
        try:
            return world.elements[np.nanargmin(distances)]
        except ValueError as e:
            return None

    def distances_to_polys(self, world):

        # distances = [self.intersect(poly)[1] for i, poly in enumerate(world.elements)]
        distances = list(all_distances(self.offset, self.direction, world.polys_to_numpy()))
        if len(world.spheres()) > 0:
            distances += [sp.closest_intersect_distance(self, return_distance=True) for sp in world.spheres()]
        distances = [d if d is not None and d > 0.00001 else np.NAN for d in distances]
        return distances

    def refract(self, geometry_element: Union['Polygon', 'Sphere'], intersect_point) -> Union[None, 'Ray']:
        # Calculate the intersection point; should handle the possibility of no intersection.
        offset = intersect_point
        if offset is None:
            return None, False

        # Calculate the normal at the intersection point.
        normal = geometry_element.normal_vector(offset)  # Assuming normal_vector takes a point.

        # Identify whether the ray is entering or exiting the medium.
        is_exiting = self.direction.dot(normal) > 0
        if is_exiting:
            # If exiting, the normal vector should be reversed.
            normal = -normal

        # Indices of refraction.
        n1 = 1.0  # Assuming air or vacuum outside the material.
        n2 = geometry_element.surface.refraction_streangth  # Assuming this is the correct property.

        # Refractive indices ratio depends on whether the ray is entering or exiting the medium.
        eta = n1 / n2 if is_exiting else n2 / n1

        # cos(theta) calculation.
        cosi = -normal.dot(self.direction)
        k = 1 - eta * eta * (1 - cosi * cosi)

        # Check for total internal reflection.
        if k < 0:
            # Total internal reflection occurred; no refraction.
            return None, True

        # Calculate refracted ray direction.
        direction = (self.direction * eta) + normal * (eta * cosi - sqrt(k))

        # Create the refracted ray.
        refracted_ray = Ray(offset, direction)

        return refracted_ray, False

    def reflect(self, geometry_element: Union['Polygon', 'Sphere'], intersect_point) -> Union[None, 'Ray']:
        if intersect_point is None:
            return None

        normal_vector = geometry_element.normal_vector(intersect_point)
        return self.reflect_formula(self.direction, intersect_point, normal_vector)

    @staticmethod
    def reflect_formula(ray_direction, intersect_point, normal_vector):
        if (-ray_direction).dot(normal_vector) < 0:
            normal_vector = -normal_vector
        # Reflect the direction around the normal
        reflected_direction = ray_direction - (normal_vector * 2 * (ray_direction.dot(normal_vector)))
        # Create a new Ray with the reflected direction
        return Ray(offset=intersect_point, direction=reflected_direction)

    def __repr__(self):
        return f"Ray({self.offset.__repr__()} + x * {self.direction.__repr__()})"


class Polygon:
    def __init__(self, a: Vector, b: Vector, c: Vector, surface: 'Surface'):
        self.a = a
        self.b = b
        self.c = c
        self.surface = surface or Surface()
        self.normal_vec = (b-a).cross(c-a).norm()

    def normal_vector(self, _):
        return self.normal_vec

    def is_inside(self, v: Vector) -> bool:
        # Compute vectors
        v0 = self.c - self.a
        v1 = self.b - self.a
        v2 = v - self.a

        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1)
        dot02 = v0.dot(v2)
        dot11 = v1.dot(v1)
        dot12 = v1.dot(v2)

        # Compute barycentric coordinates
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v < 1)

    def transpose(self, vec: Vector) -> 'Polygon':
        return Polygon(self.a + vec, self.b + vec, self.c + vec, self.surface)

    def rotate(self, angle: float, axis: 'Ray'):
        return Polygon(self.a.rotate(angle, axis), self.b.rotate(angle, axis), self.c.rotate(angle, axis), self.surface)

    def __eq__(self, other):
        return {self.a, self.b, self.c} == {other.a, other.b, other.c}

    def __hash__(self):
        return self.a.round(5).__hash__() + self.b.round(5).__hash__() + self.c.round(5).__hash__()

    def __repr__(self):
        return f"Polygon: {self.a.__repr__()} {self.b.__repr__()} {self.c.__repr__()}"

    def vectors(self):
        return [self.a, self.b, self.c]


class Geometry:
    elements: List[Union[Polygon, 'Sphere']]
    polys_np = None
    def __init__(self, *polys: Polygon):
        self.elements = list(polys)

    def __add__(self, other: 'Geometry'):
        return Geometry(*(self.elements + other.elements))

    def transpose(self, vec: Vector) -> 'Geometry':
        return Geometry(*[poly.transpose(vec) for poly in self.elements])

    def rotate(self, angle: float, axis: 'Ray'):
        return Geometry(*[poly.rotate(angle, axis) for poly in self.elements])

    def __eq__(self, other):
        return set([vec for element in self.elements for vec in element.vectors()]) \
               == set([vec for element in other.elements for vec in element.vectors()])

    def __hash__(self):
        return sum([el.__hash__() for el in self.elements])

    def __repr__(self):
        return "Geometry: " + " ".join({vec.round(5).__repr__() for poly in self.elements for vec in poly.vectors()})

    def polys_to_numpy(self):
        if self.polys_np is None:
            self.polys_np=np.array([np.array([el.a, el.b, el.c, el.normal_vec])
                                    for el in self.elements if type(el) == Polygon]).astype(np.float64)
        return self.polys_np

    def spheres_to_numpy(self):
        if not self.spheres_np:
            self.spheres_np=np.array([np.array([el.pos, el.radius])
                                    for el in self.elements if type(el) == Sphere])
            return self.polys_np

    def spheres(self):
        return [el for el in self.elements if type(el) == Sphere]


class Square(Geometry):
    def __init__(self, a: Vector, b: Vector, c: Vector, d: Vector, surface: 'Surface'):
        super().__init__(*[Polygon(a, b, c, surface), Polygon(b, c, d, surface)])


class Cube(Geometry):
    def __init__(self, base: Vector, width: float, surface: 'Surface'):
        bottom = Square(
            base,
            base + Vector.from_list([width, 0, 0]),
            base + Vector.from_list([0, width, 0]),
            base + Vector.from_list([width, width, 0]), surface)
        top = bottom.transpose(Vector.from_list([0, 0, width]))

        front = Square(
            base,
            base + Vector.from_list([width, 0, 0]),
            base + Vector.from_list([0, 0, width]),
            base + Vector.from_list([width, 0, width]), surface)
        back = front.transpose(Vector.from_list([0, width, 0]))

        left = Square(
            base,
            base + Vector.from_list([0, width, 0]),
            base + Vector.from_list([0, 0, width]),
            base + Vector.from_list([0, width, width]), surface)
        right = left.transpose(Vector.from_list([width, 0, 0]))

        super().__init__(*(bottom + top + left + right + back + front).elements)

    @classmethod
    def from_center(cls, center: Vector, width: float, surface: 'Surface'):
        base = center - Vector.from_list([width/2, width/2, width/2])
        return cls(base, width, surface)


class Sphere:
    def __init__(self, pos: Vector, radius: float, surface: 'Surface'):
        self.pos = pos
        self.radius = radius
        self.surface = surface

    def closest_intersect_distance(self, ray: Ray, return_distance: bool = False) -> Optional[Union[Vector, float]]:
        """ Will always return the closest intersect """
        projection = ray.direction.dot(ray.offset - self.pos)
        root_content = pow(projection, 2) - (pow((ray.offset - self.pos).abs(), 2) - self.radius*self.radius)
        if root_content < 0:
            return None
        first = -projection - sqrt(root_content)
        second = -projection + sqrt(root_content)
        if first > 0.0001 and second > 0.0001:
            intersect_point_on_ray = min(first, second)
        elif first > 0.0001:
            intersect_point_on_ray = first
        elif second > 0.0001:
            intersect_point_on_ray = second
        else:
            return None
        if intersect_point_on_ray:
            intersect_vector, distance = ray.point_on_line(intersect_point_on_ray)
            if return_distance:
                return distance
            else:
                return intersect_vector
        else:
            return None

    def normal_vector(self, intersect: Vector):
        return (intersect - self.pos).norm()


class Surface:
    def __init__(self, k_ambient=0, k_diffuse=1, k_reflect=0, color: Vector=None, k_refraction: float = 0,
                 refraction_streangth: float = 1.33):
        self.k_ambient = k_ambient
        self.k_diffuse = k_diffuse
        self.k_reflect = k_reflect
        self.color = color if color is not None else Vector.from_list([1, 1, 1])
        self.k_refraction = k_refraction
        self.refraction_streangth = refraction_streangth
