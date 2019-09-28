from polliwog.tri import shapes


def _shape_as_mesh(shape_factory_fn, *args, **kwargs):
    from lace.mesh import Mesh
    vertices, faces = shape_factory_fn(*args, ret_unique_vertices_and_faces=True, **kwargs)
    return Mesh(v=vertices, f=faces)


def create_rectangular_prism(*args, **kwargs):
    '''
    Return a Mesh which is an axis-aligned rectangular prism. One vertex is
    `origin`; the diametrically opposite vertex is `origin + size`.

    size: 3x1 array.

    '''
    return _shape_as_mesh(shapes.create_rectangular_prism, *args, **kwargs)


def create_cube(*args, **kwargs):
    '''
    Return a mesh with an axis-aligned cube. One vertex is `origin`; the
    diametrically opposite vertex is `size` units along +x, +y, and +z.

    size: int or float.

    '''
    return _shape_as_mesh(shapes.create_cube, *args, **kwargs)


def create_triangular_prism(*args, **kwargs):
    '''
    Return a Mesh which is a triangular prism whose base is the triangle
    p1, p2, p3. If the vertices are oriented in a counterclockwise
    direction, the prism extends from behind them.

    '''
    return _shape_as_mesh(shapes.create_triangular_prism, *args, **kwargs)


def create_horizontal_plane(*args, **kwargs):
    '''
    Creates a horizontal plane.
    '''
    return _shape_as_mesh(shapes.create_horizontal_plane, *args, **kwargs)


def _main():
    import math
    import numpy as np

    points = np.array([
        [1, 0, 0],
        [0, math.sqrt(1.25), 0],
        [-1, 0, 0],
    ])
    prism = create_triangular_prism(*points, height=4)
    prism.show()

    cube = create_cube(np.array([1., 0., 0.]), 4.)
    cube.show()


if __name__ == '__main__':
    _main()
