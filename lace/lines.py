import numpy as np
from blmath.numerics import as_numeric_array
from blmath.util.decorators import setter_property

__all__ = ['Lines']

class Lines(object):
    """3d List-of-lines

    Attributes:
        v: Vx3 array of vertices
        e: Ex2 array of edges
    """

    def __init__(self, v, e, vc=None, ec=None):
        self.v = v
        self.e = e
        self.vc = vc
        self.ec = ec

    @setter_property
    def v(self, val):  # pylint: disable=method-hidden
        self.__dict__['v'] = as_numeric_array(val, dtype=np.float64, shape=(-1, 3), allow_none=True, empty_as_none=True)

    @setter_property
    def e(self, val):  # pylint: disable=method-hidden
        self.__dict__['e'] = as_numeric_array(val, dtype=np.uint64, shape=(-1, 2), allow_none=True, empty_as_none=True)

    @setter_property
    def vc(self, val):  # pylint: disable=method-hidden
        from lace import color
        val = color.colors_like(val, self.v)
        self.__dict__['vc'] = val

    @setter_property
    def ec(self, val):  # pylint: disable=method-hidden
        from lace import color
        val = color.colors_like(val, self.e)
        self.__dict__['ec'] = val

    def write_obj(self, filename):
        with open(filename, 'w') as fi:
            for r in self.v:
                fi.write('v %f %f %f\n' % (r[0], r[1], r[2]))
            for e in self.e:
                fi.write('l %d %d\n' % (e[0]+1, e[1]+1))

    def show(self, mv=None):
        from lace.meshviewer import MeshViewer

        if mv is None:
            mv = MeshViewer(keepalive=True)

        mv.set_dynamic_lines([self])

        return mv

    def all_edges_with_verts(self, v_indices, as_boolean=False):
        '''
        returns all of the faces that contain at least one of the vertices in v_indices
        '''
        included_vertices = np.zeros(self.v.shape[0], dtype=bool)
        included_vertices[v_indices] = True
        edges_with_verts = included_vertices[self.e].all(axis=1)
        if as_boolean:
            return edges_with_verts
        return np.nonzero(edges_with_verts)[0]

    def keep_vertices(self, indices_to_keep, ret_kept_edges=False):
        '''
        Keep the given vertices and discard the others, and any edges to which
        they may belong.


        If `ret_kept_edges` is `True`, return the original indices of the kept
        edges. Otherwise return `self` for chaining.

        '''

        if self.v is None:
            return

        initial_num_verts = self.v.shape[0]
        if self.e is not None:
            initial_num_edges = self.e.shape[0]
            e_indices_to_keep = self.all_edges_with_verts(indices_to_keep, as_boolean=True)

        self.v = self.v[indices_to_keep]
        if self.e is not None:
            v_old_to_new = np.zeros(initial_num_verts, dtype=int)
            e_old_to_new = np.zeros(initial_num_edges, dtype=int)

            v_old_to_new[indices_to_keep] = np.arange(len(indices_to_keep), dtype=int)
            self.e = v_old_to_new[self.e[e_indices_to_keep]]
            e_old_to_new[e_indices_to_keep] = np.arange(self.e.shape[0], dtype=int)
        else:
            e_indices_to_keep = []

        return np.nonzero(e_indices_to_keep)[0] if ret_kept_edges else self

    def remove_vertices(self, v_list):
        return self.keep_vertices(np.setdiff1d(np.arange(self.v.shape[0]), v_list))
