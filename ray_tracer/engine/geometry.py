import numpy as np
from typing import Union, List, Tuple, Optional
from scipy.spatial.transform import Rotation
from math import acos


class Vector:
    def __init__(self, vec: Union[List[float], np.array]):
        self.vec = np.array(vec).reshape((3))

    def distance(self, other: 'Vector') -> 'Vector':
        return (self - other).abs()

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector([self.vec[1]*other.vec[2]-self.vec[2]*other.vec[1],
                       self.vec[2]*other.vec[0]-self.vec[0]*other.vec[2],
                       self.vec[0]*other.vec[1]-self.vec[1]*other.vec[0]])

    def dot(self, other: 'Vector') -> float:
        return np.dot(self.vec.T, other.vec)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.vec + other.vec)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.vec - other.vec)

    def __floordiv__(self, other: float) -> 'Vector':
        return Vector(self.vec / other)

    def __truediv__(self, other: float) -> 'Vector':
        return Vector(self.vec / other)

    def __mul__(self, other: float) -> 'Vector':
        return Vector(self.vec * other)

    def __eq__(self, other: 'Vector') -> bool:
        return np.array_equal(self.vec, other.vec)

    def __str__(self) -> str:
        return f"Vector({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return int(self.vec.sum())

    def abs(self):
        return np.linalg.norm(self.vec)

    def norm(self) -> 'Vector':
        return self / self.abs()

    def rotate(self, angle: float, axis: 'Ray'):

        rotation_radians = np.radians(angle)
        rotation_axis = axis.direction.norm()
        rotation_vector = rotation_axis * rotation_radians
        rotation = Rotation.from_rotvec(rotation_vector.vec)
        rotated_vec = Vector(rotation.apply((self - axis.offset).vec)) + axis.offset
        return rotated_vec

    def round(self, decimals: int):
        return Vector(self.vec.round(decimals))

    def angle(self, other: 'Vector'):
        return (acos(self.dot(other) / (self.abs()*other.abs())))*(360/(2*np.pi))


class Ray:
    def __init__(self, offset: Vector, direction: Vector):
        self.offset = offset
        self.direction = direction

    def point_on_line(self, scaler: float) -> Tuple['Vector', float]:
        return self.offset + self.direction * scaler, scaler

    def offset_distance(self, point: Vector):
        self.offset.distance(point)

    def reflect(self, poly) -> Union[None, 'Ray']:
        offset = self.intersect(poly)[0]

        directed_normal = poly.normal_vector if self.direction.angle(poly.normal_vector) > 90 else poly.normal_vector*-1

        if offset is None:
            return None
        direction = self.direction - directed_normal*(self.direction.dot(directed_normal)*2)
        return Ray(offset=self.intersect(poly)[0], direction=direction)

    def intersect(self, poly: 'Polygon') -> Tuple[Optional[Vector], Optional[float]]:
        intersect, distance = self.point_on_line(
            (poly.a - self.offset).dot(poly.normal_vector) / (self.direction.dot(poly.normal_vector))
        )
        return (None, None) if not poly.is_inside(intersect) else (intersect, distance)

    def multi_reflect(self, world: 'Geometry', distances: List[float]) -> Union[None, 'Ray']:
        try:
            first_intersect = np.nanargmin(distances)
        except ValueError as e:
            return None
        return self.reflect(world.elements[first_intersect])

    def distances_to_polys(self, world):
        distances = [self.intersect(poly)[1] for i, poly in enumerate(world.elements)]
        distances = [d if d is not None and d > 0.00001 else np.NAN for d in distances]
        return distances

    def __repr__(self):
        return f"Ray({self.offset.__repr__()} + x * {self.direction.__repr__()})"


class Polygon:
    def __init__(self, a: Vector, b: Vector, c: Vector):
        self.a = a
        self.b = b
        self.c = c
        self.normal_vector = (b-a).cross(c-a).norm()

    def is_inside(self, v: Vector) -> bool:
        area = (self.b - self.a).cross(self.c-self.a).abs()/2
        alpha = (self.b - v).cross(self.c - v).abs() / (2*area)
        beta = (self.c - v).cross(self.a - v).abs() / (2*area)
        gamma = 1 - alpha - beta
        return (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1)

    def transpose(self, vec: Vector) -> 'Polygon':
        return Polygon(self.a + vec, self.b + vec, self.c + vec)

    def rotate(self, angle: float, axis: 'Ray'):
        return Polygon(self.a.rotate(angle, axis), self.b.rotate(angle, axis), self.c.rotate(angle, axis))

    def __eq__(self, other):
        return {self.a, self.b, self.c} == {other.a, other.b, other.c}

    def __repr__(self):
        return f"Polygon: {self.a.__repr__()} {self.b.__repr__()} {self.c.__repr__()}"

    def vectors(self):
        return [self.a, self.b, self.c]


class Geometry:
    elements: List[Polygon]

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


class Square(Geometry):
    def __init__(self, a: Vector, b: Vector, c: Vector, d: Vector):
        super().__init__(*[Polygon(a, b, c), Polygon(b, c, d)])


class Cube(Geometry):
    def __init__(self, base: Vector, width: float):
        bottom = Square(
            base,
            base + Vector([width, 0, 0]),
            base + Vector([0, width, 0]),
            base + Vector([width, width, 0]))
        top = bottom.transpose(Vector([0, 0, width]))

        front = Square(
            base,
            base + Vector([width, 0, 0]),
            base + Vector([0, 0, width]),
            base + Vector([width, 0, width]))
        back = front.transpose(Vector([0, width, 0]))

        left = Square(
            base,
            base + Vector([0, width, 0]),
            base + Vector([0, 0, width]),
            base + Vector([0, width, width]))
        right = left.transpose(Vector([width, 0, 0]))

        super().__init__(*(bottom + top + left + right + back + front).elements)

    @classmethod
    def from_center(cls, center: Vector, width: float):
        base = center - Vector([width/2, width/2, width/2])
        return cls(base, width)
