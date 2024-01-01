from pathlib import Path

from stl import mesh
from ray_tracer.engine.geometry import Polygon, Vector, Surface, Geometry, Ray

def stl_to_polys(position: Vector, path: Path, surface: Surface):
    data = mesh.Mesh.from_file(path)
    poly_vertices = data.vectors.reshape(-1, 3, 3)

    polys = []
    for i in range(poly_vertices.shape[0]):
        stl_vectors = [position + Vector(poly_vertices[i, j, :]) for j in range(3)]

        poly_vectors = [position + Vector(poly_vertices[i, j, :]) for j in range(3)]
        polys.append(Polygon(*poly_vectors, surface=surface))

    geometry = Geometry(*polys)
    geometry = geometry.rotate(-90, Ray(position, Vector.from_list([0, 1, 0])))

    return geometry


if __name__ == "__main__":
    stl_to_polys(Vector.from_list([0, 0, 0]), Path(__file__).parent.parent / "resources" / "rabbit.stl", surface=None)