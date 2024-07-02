#################################################
# This file contains utility functions that are #
# used in the main script.                      #
#################################################
from icecream import ic

def linear_interpolation(x, x0, y0, x1, y1):
    """
    Performs linear interpolation to estimate the value of y at a given x
    based on two known points (x0, y0) and (x1, y1).
    """
    if x0 == x1:
        raise ValueError("x0 and x1 must be different")

    # Calculate the slope
    slope = (y1 - y0) / (x1 - x0)

    # Calculate the y-intercept
    intercept = y0 - slope * x0

    # Calculate the estimated y value
    y = slope * x + intercept

    return y

def bilinear_interpolation(x, y, x0, y0, x1, y1, q00, q01, q10, q11):
    """
    Performs bilinear interpolation to estimate the value of z at a given (x, y)
    based on the values of q at the four corners (x0, y0), (x0, y1), (x1, y0), and (x1, y1).
    """
    if x0 == x1 or y0 == y1:
        raise ValueError("x0 and x1, and y0 and y1, must be different")

    # Perform linear interpolation in the x-direction
    r0 = linear_interpolation(x, x0, q00, x1, q10)
    r1 = linear_interpolation(x, x0, q01, x1, q11)

    # Perform linear interpolation in the y-direction
    z = linear_interpolation(y, y0, r0, y1, r1)

    return z

# time function by decorators
import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} took: {te-ts} sec\n')
        return result
    return timed

# import numpy as np
# def find_3d_point(x: int, y: int, vertex):
#     l = list(vertex.data[y * 2048 + x])
#     return np.array(l[:3])

# def get_3dcoord_bilinear(x:float, y:float, vertex):
#     x0, y0 = int(x), int(y)
#     x1, y1 = x0 + 1, y0 + 1

#     q00 = find_3d_point(x0, y0, vertex)
#     q01 = find_3d_point(x0, y1, vertex)
#     q10 = find_3d_point(x1, y0, vertex)
#     q11 = find_3d_point(x1, y1, vertex)

#     z = bilinear_interpolation(x, y, x0, y0, x1, y1, q00, q01, q10, q11)
#     return z
