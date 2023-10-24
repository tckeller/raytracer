import numpy as np

def all_distances(offset, direction, world):

    distances = np.zeros(world.shape[0])
    for i in range(world.shape[0]):
        distances[i] = intersect(offset, direction, world[i])
    return distances

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def point_on_line(offset, direction, scaler):
    return offset + direction * scaler

def intersect(offset, direction, poly):
    projection = dot(direction,poly[3])

    if projection == 0:
        return -1.0

    distance = dot((poly[0] - offset),poly[3]) / projection
    intersect = point_on_line(offset, direction, distance)
    if not is_inside(poly[0:3], intersect):
        return -1.0
    else:
        return distance

def is_inside(poly, v):
    # Compute vectors

    v0 = poly[2] - poly[0]
    v1 = poly[1] - poly[0]
    v2 = v - poly[0]

    dot00 = dot(v0,v0)
    dot01 = dot(v0,v1)
    dot02 = dot(v0,v2)
    dot11 = dot(v1,v1)
    dot12 = dot(v1,v2)

    # Compute barycentric coordinates
    d_sum = dot00 * dot11 - dot01 * dot01
    if d_sum == 0:
        return False
    else:
        invDenom = 1 / d_sum
        uu = (dot11 * dot02 - dot01 * dot12) * invDenom
        vv = (dot00 * dot12 - dot01 * dot02) * invDenom

        # Check if point is in triangle
        return (uu >= 0) and (vv >= 0) and (uu + vv <= 1)