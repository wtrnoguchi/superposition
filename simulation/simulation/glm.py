import math

import numpy


def identity():
    return numpy.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], numpy.float32
    )


def scale(x, y, z):
    return numpy.matrix(
        [
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], numpy.float32
    )


def translation(x, y, z):
    return numpy.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ], numpy.float32
    )


def rotation(angDeg, x, y, z):

    angRad = math.radians(angDeg)
    s = math.sin(-angRad)
    c = math.cos(-angRad)
    mag = math.sqrt(x * x + y * y + z * z)

    x /= mag
    y /= mag
    z /= mag

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    zx = z * x
    xs = x * s
    ys = y * s
    zs = z * s
    one_c = 1.0 - c

    return numpy.matrix(
        [
            [(one_c * xx) + c, (one_c * xy) - zs, (one_c * zx) + ys, 0.0], [
                (one_c * xy) + zs, (one_c * yy) + c, (one_c * yz) - xs, 0.0
            ], [(one_c * zx) - ys, (one_c * yz) + xs,
                (one_c * zz) + c, 0.0], [0.0, 0.0, 0.0, 1.0]
        ], numpy.float32
    )


def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2.0)

    return numpy.matrix(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (near + far) / (near - far), -1.0],
            [0.0, 0.0, 2.0 * near * far / (near - far), 0.0],
        ], numpy.float32
    )


def orthogonal(t, b, r, l, near, far):
    top = float(t)
    bottom = float(b)
    right = float(r)
    left = float(l)

    return numpy.matrix(
        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [
                -(right + left) / (right - left), -(top + bottom) /
                (top - bottom), -(far + near) / (far - near), 1.0
            ],
        ], numpy.float32
    )
