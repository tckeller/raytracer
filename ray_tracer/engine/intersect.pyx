from libcpp cimport bool

cimport numpy as np
import numpy as npply
cimport cython
from cython.view cimport array as cvarray
np.import_array()

# Import statements here
from libc.math cimport sqrt

cpdef np.ndarray[np.float64_t] all_distances(np.ndarray[np.float64_t] offset, np.ndarray[np.float64_t] direction,
                                            np.ndarray[np.float64_t, ndim = 3] world):
    cdef int i
    cdef np.ndarray[np.float64_t] distances
    cdef np.npy_intp dim = 0


    distances = npply.zeros(world.shape[0], dtype=npply.float64)
    for i in range(world.shape[0]):
        distances[i] = intersect(offset, direction, world[i])
    return distances


# Use this cdef for functions you don't want to expose to Python
cdef double dot(double[:] vec1, double[:] vec2):
    cdef int i
    cdef double result = 0.0
    for i in range(vec1.shape[0]):
        result += vec1[i] * vec2[i]
    return result

cdef double[:] point_on_line(double[:] point, double[:] direction, double distance):
    # Allocate a Cython array for the result. This supports conversion to memory views.
    cdef double[3] result_arr  # declare a static array instead
    cdef int i
    for i in range(3):  # Assuming 3D vectors
        result_arr[i] = point[i] + distance * direction[i]

    # Create a Cython array from the C array. This array is heap-allocated and can be returned safely.
    cdef double[:] result_view = cvarray(shape=(3,), itemsize=sizeof(double), format="d")
    for i in range(3):
        result_view[i] = result_arr[i]

    return result_view

cdef void subtract_vectors(double[:] vec1, double[:] vec2, double[:] result) nogil:
    cdef int i
    for i in range(vec1.shape[0]):
        result[i] = vec1[i] - vec2[i]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate division by zero checking in C division
def ray_intersects_aabb(double[:] ray_origin, double[:] ray_direction, double[:] aabb_min, double[:] aabb_max):
    cdef double t1, t2, t3, t4, t5, t6, tmin, tmax
    cdef double dir_fraction[3]

    # Preventing division by zero and costly operations by calculating inverses once
    for i in range(3):
        dir_fraction[i] = 1.0 / ray_direction[i] if ray_direction[i] != 0 else float('inf')

    t1 = (aabb_min[0] - ray_origin[0]) * dir_fraction[0]
    t2 = (aabb_max[0] - ray_origin[0]) * dir_fraction[0]
    t3 = (aabb_min[1] - ray_origin[1]) * dir_fraction[1]
    t4 = (aabb_max[1] - ray_origin[1]) * dir_fraction[1]
    t5 = (aabb_min[2] - ray_origin[2]) * dir_fraction[2]
    t6 = (aabb_max[2] - ray_origin[2]) * dir_fraction[2]

    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

    # Conditions for a valid intersection
    if tmax < 0 or tmin > tmax:
        return False

    return True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate division by zero checking in C division
cpdef double intersect(double[:] offset, double[:] direction, double[:, :] poly):
    cdef:
        double[:] intersect_point  # Changed to a memoryview
        double[:] temp_subtract_result = npply.empty(3, dtype=npply.float64)
        double distance, projection

    if not ray_intersects_aabb(offset, direction, poly[4], poly[5]):
        return -1.0

    projection = dot(direction, poly[3])

    if projection == 0:
        return -1.0

    # Correctly handle vector subtraction.
    subtract_vectors(poly[0], offset, temp_subtract_result)
    distance = dot(temp_subtract_result, poly[3]) / projection

    # Create the point inline to avoid function call overhead
    intersect_point = point_on_line(offset, direction, distance)

    if not is_inside(poly[0:3], intersect_point):
        return -1.0
    else:
        return distance

cdef bint is_inside(double[:, :] poly, double[:] point):
    cdef:
        double[3] v0, v1, v2  # Static size arrays for speed.
        double dot00, dot01, dot02, dot11, dot12, invDenom, u, v
        int i

    # Optimize subtraction operation here
    for i in range(3):
        v0[i] = poly[2, i] - poly[0, i]
        v1[i] = poly[1, i] - poly[0, i]
        v2[i] = point[i] - poly[0, i]

    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)

    # Avoid repeated calculation by computing the inverse of the denominator.
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v < 1)