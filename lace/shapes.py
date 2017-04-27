import math
import numpy as np
from lace.mesh import Mesh


def create_rectangular_prism(origin, size):
    '''
    Return a Mesh which is an axis-aligned rectangular prism. One vertex is
    `origin`; the diametrically opposite vertex is `origin + size`.

    size: 3x1 array.

    '''
    from lace.topology import quads_to_tris

    lower_base_plane = np.array([
        # Lower base plane
        origin,
        origin + np.array([size[0], 0, 0]),
        origin + np.array([size[0], 0, size[2]]),
        origin + np.array([0, 0, size[2]]),
    ])
    upper_base_plane = lower_base_plane + np.array([0, size[1], 0])

    vertices = np.vstack([lower_base_plane, upper_base_plane])

    faces = quads_to_tris(np.array([
        [0, 1, 2, 3],  # lower base (-y)
        [7, 6, 5, 4],  # upper base (+y)
        [4, 5, 1, 0],  # +z face
        [5, 6, 2, 1],  # +x face
        [6, 7, 3, 2],  # -z face
        [3, 7, 4, 0],  # -x face
    ]))

    return Mesh(v=vertices, f=faces)


def create_cube(origin, size):
    '''
    Return a mesh with an axis-aligned cube. One vertex is `origin`; the
    diametrically opposite vertex is `size` units along +x, +y, and +z.

    size: int or float.

    '''
    return create_rectangular_prism(origin, np.repeat(size, 3))


def create_triangular_prism(p1, p2, p3, height):
    '''
    Return a Mesh which is a triangular prism whose base is the triangle
    p1, p2, p3. If the vertices are oriented in a counterclockwise
    direction, the prism extends from behind them.

    '''
    from blmath.geometry import Plane

    base_plane = Plane.from_points(p1, p2, p3)
    lower_base_to_upper_base = height * -base_plane.normal # pylint: disable=invalid-unary-operand-type
    vertices = np.vstack(([p1, p2, p3], [p1, p2, p3] + lower_base_to_upper_base))

    faces = np.array([
        [0, 1, 2],  # base
        [0, 3, 4], [0, 4, 1],  # side 0, 3, 4, 1
        [1, 4, 5], [1, 5, 2],  # side 1, 4, 5, 2
        [2, 5, 3], [2, 3, 0],  # side 2, 5, 3, 0
        [5, 4, 3],  # base
    ])

    return Mesh(v=vertices, f=faces)


def create_horizontal_plane():
    '''
    Creates a horizontal plane.
    '''
    v = np.array([
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 0., 1.],
        [0., 0., -1.]
    ])
    f = [[0, 1, 2], [3, 1, 0]]
    return Mesh(v=v, f=f)


if __name__ == '__main__':
    points = np.array([
        [1, 0, 0],
        [0, math.sqrt(1.25), 0],
        [-1, 0, 0],
    ])
    prism = create_triangular_prism(*points, height=4)
    prism.show()

    cube = create_cube(np.array([1., 0., 0.]), 4.)
    cube.show()
