from libcpp cimport bool

cimport numpy as np
import numpy as npply
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

cpdef double intersect(double[:] offset, double[:] direction, double[:, :] poly):
    cdef:
        double[:] intersect_point  # Changed to a memoryview
        double[:] temp_subtract_result = npply.empty(3, dtype=npply.float64)
        double distance, projection

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