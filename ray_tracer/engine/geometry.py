import numpy
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
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


    def refract(self, geometry_element: Union['Polygon', 'Sphere'], intersect_point: Vector, is_inside: bool) -> Tuple[Union[None, 'Ray'], bool]:
        """
        Calculate the refracted ray at the intersection point on the geometry.

        :param geometry_element: The geometric object being intersected.
        :param intersect_point: The point of intersection.
        :param is_inside: If the ray is currently inside the medium or outside. This will determine the refraction direction.
        :return: A tuple of the refracted ray (or None if total internal reflection occurs) and a boolean indicating whether total internal reflection occurred.
        """
        if intersect_point is None:
            return None, False

        # Calculate the normal at the intersection point.
        normal = geometry_element.normal_vector(intersect_point)
        incoming_dir = self.direction

        # Calculate cos(theta_i) using the absolute value of the dot product.

        cosi = normal.dot(incoming_dir)

        if cosi < 0:
            normal = -normal
            cosi = normal.dot(incoming_dir)

        # Define the refractive indices for air (or vacuum) and the material.
        n2 = geometry_element.surface.refraction_strength  # Index for the material.

        # Calculate the ratio of the refractive indices, considering the direction of the ray.
        eta = 1 / n2 if not is_inside else n2 / 1

        # Calculate sin^2(theta_t).
        sint2 = eta ** 2 * (1.0 - cosi ** 2)

        # Handle total internal reflection.
        if sint2 > 1.0:
            return None, True  # Total internal reflection occurred, no refraction

        # Calculate cos(theta_t) using trigonometric identity.
        cost = np.sqrt(1.0 - sint2)

        # Calculate the direction of the refracted ray.
        refract_dir = eta * incoming_dir + (eta * cosi - cost) * normal
        refract_dir = refract_dir / np.linalg.norm(refract_dir)  # Normalize the direction.

        # Create the refracted ray, ensuring it starts from the intersection point going in the refracted direction.
        refracted_ray = Ray(intersect_point, refract_dir)
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
        self.bbox_top, self.bbox_bottom =self.create_bounding_box(a, b, c)

    @staticmethod
    def create_bounding_box(a, b, c):
        # Unzip the list of vertices into three separate lists for x, y, and z coordinates.
        xs, ys, zs = zip(*[a, b, c])

        # Find the minimum and maximum values for each dimension.
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        # Define the bottom-left-front and top-right-back corners of the bounding box.
        bottom_left_front = (min_x, min_y, min_z)
        top_right_back = (max_x, max_y, max_z)

        return bottom_left_front, top_right_back

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
    spheres_np = None
    def __init__(self, *polys: Union[Polygon, 'Sphere']):
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
            self.polys_np=np.array([np.array([el.a, el.b, el.c, el.normal_vec, el.bbox_top, el.bbox_bottom])
                                    for el in self.elements if type(el) == Polygon]).astype(np.float64)
        return self.polys_np

    def spheres_to_numpy(self):
        if not self.spheres_np:
            self.spheres_np=np.array([np.array([el.pos, el.radius])
                                    for el in self.elements if type(el) == Sphere])
            return self.spheres_np

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

    def transpose(self, vec: Vector) -> 'Sphere':
        return Sphere(pos=self.pos + vec, radius=self.radius, surface=self.surface)

    def rotate(self, angle: float, axis: 'Ray') -> 'Sphere':
        return Sphere(pos=self.pos.rotate(angle, axis), radius=self.radius, surface=self.surface)


class Surface:
    def __init__(self, k_ambient=0, k_diffuse=1, k_reflect=0, color: Vector=None, k_refraction: float = 0,
                 refraction_strength: float = 1.33, colormap: Optional[Callable[[Vector], Vector]] = None):
        self.k_ambient = k_ambient
        self.k_diffuse = k_diffuse
        self.k_reflect = k_reflect
        self._color = color if color is not None else Vector.from_list([1, 1, 1])
        self.colormap = colormap
        self.k_refraction = k_refraction
        self.refraction_strength = refraction_strength

    def color(self, position: Vector):
        if self.colormap is None:
            return self._color
        else:
            return self.colormap(position)
