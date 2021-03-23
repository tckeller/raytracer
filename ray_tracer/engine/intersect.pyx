from libcpp cimport bool

cimport numpy as np
import numpy as npply
np.import_array()

cpdef np.ndarray[np.double_t] all_distances(np.ndarray[np.double_t] offset, np.ndarray[np.double_t] direction,
                                            np.ndarray[np.double_t, ndim = 3] world):
    cdef int i
    cdef np.ndarray[np.double_t] distances
    cdef np.npy_intp dim = 0


    distances = npply.zeros(world.shape[0])
    for i in range(world.shape[0]):
        distances[i] = intersect(offset, direction, world[i])
    return distances

cdef double dot(np.ndarray[np.double_t] a, np.ndarray[np.double_t] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cdef np.ndarray[np.double_t] point_on_line(np.ndarray[np.double_t] offset, np.ndarray[np.double_t] direction, double scaler):
    return offset + direction * scaler

cpdef double intersect(np.ndarray[np.double_t] offset, np.ndarray[np.double_t] direction,
                          np.ndarray[np.double_t, ndim=2] poly):
    cdef np.ndarray[np.double_t] intersect
    cdef double distance, projection

    projection = dot(direction,poly[3])

    if projection == 0:
        return -1.0

    distance = dot((poly[0] - offset),poly[3]) / projection
    intersect = point_on_line(offset, direction, distance)
    if not is_inside(poly[0:3], intersect):
        return -1.0
    else:
        return distance

cdef int is_inside(np.ndarray[np.double_t, ndim=2] poly, np.ndarray[np.double_t] v):
    # Compute vectors
    cdef np.ndarray[np.double_t] v0, v1, v2
    cdef double dot00, dot01, dot02, dot11, dot12, invDenom, uu, vv

    v0 = poly[2] - poly[0]
    v1 = poly[1] - poly[0]
    v2 = v - poly[0]

    dot00 = dot(v0,v0)
    dot01 = dot(v0,v1)
    dot02 = dot(v0,v2)
    dot11 = dot(v1,v1)
    dot12 = dot(v1,v2)

    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    uu = (dot11 * dot02 - dot01 * dot12) * invDenom
    vv = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (uu >= 0) and (vv >= 0) and (uu + vv < 1)