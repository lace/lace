import numpy as np
import vg

class _EdgeMap(object):
    """
    A quick two-level dictionary where the two keys are interchangeable (i.e.
    a symmetric graph).
    """
    def __init__(self):
        self.d = {} # store indicies into self.values here, to make it easier to get inds or values
        self.values = []
    def _order(self, u, v):
        if u < v:
            return u, v
        else:
            return v, u
    def add(self, u, v, val):
        low, high = self._order(u, v)
        if low not in self.d:
            self.d[low] = {}
        self.values.append(val)
        self.d[low][high] = len(self.values) - 1
    def contains(self, u, v):
        low, high = self._order(u, v)
        if low in self.d and high in self.d[low]:
            return True
        return False
    def index(self, u, v):
        low, high = self._order(u, v)
        try:
            return self.d[low][high]
        except KeyError:
            return None
    def get(self, u, v):
        ii = self.index(u, v)
        if ii is not None:
            return self.values[ii]
        else:
            return None


class _Graph(object):
    """
    A little utility class to build a symmetric graph and calculate Euler Paths.
    """
    def __init__(self, size):
        self.size = size
        self.d = {}
    def __len__(self):
        return len(self.d)
    def add_edges(self, edges):
        for u, v in edges:
            self.add_edge(u, v)
    def add_edge(self, u, v):
        assert u >= 0 and u < self.size
        assert v >= 0 and v < self.size
        if u not in self.d:
            self.d[u] = set()
        if v not in self.d:
            self.d[v] = set()
        self.d[u].add(v)
        self.d[v].add(u)
    def remove_edge(self, u, v):
        if u in self.d and v in self.d[u]:
            self.d[u].remove(v)
        if v in self.d and u in self.d[v]:
            self.d[v].remove(u)
        if v in self.d and len(self.d[v]) == 0:
            del self.d[v]
        if u in self.d and len(self.d[u]) == 0:
            del self.d[u]
    def pop_euler_path(self, allow_multiple_connected_components=True):
        # Based on code from Przemek Drochomirecki, Krakow, 5 Nov 2006
        # http://code.activestate.com/recipes/498243-finding-eulerian-path-in-undirected-graph/
        # Under PSF License
        # NB: MUTATES d

        # counting the number of vertices with odd degree
        odd = [x for x in self.d if len(self.d[x])&1]
        odd.append(list(self.d.keys())[0])
        if not allow_multiple_connected_components and len(odd) > 3:
            return None
        stack = [odd[0]]
        path = []
        # main algorithm
        while stack:
            v = stack[-1]
            if v in self.d:
                u = self.d[v].pop()
                stack.append(u)
                self.remove_edge(u, v)
            else:
                path.append(stack.pop())
        return path


class MeshMixin(object):
    def faces_intersecting_plane(self, plane):
        sgn_dists = plane.signed_distance(self.v)
        which_fs = np.abs(np.sign(sgn_dists)[self.f].sum(axis=1)) != 3
        return self.f[which_fs]

    def intersect_plane(self, plane, neighborhood=None, ret_pointcloud=False):
        '''
        Takes a cross section of planar point cloud with a Mesh object.
        Ignore those points which intersect at a vertex - the probability of
        this event is small, and accounting for it complicates the algorithm.

        If 'neighborhood' is provided, use a KDTree to constrain the
        cross section to the closest connected component to 'neighborhood'.

        When `ret_pointcloud` is true, return an unstructured point cloud
        instead of polyline(s). This is useful when (1) you only care about
        e.g. some apex of the intersection and (2) you are not specifying
        a neighborhood.

        Params:
            - plane:
                polliwog.Plane object
            - neigbhorhood:
                M x 3 np.array
            - ret_pointcloud:
                When `True`, return an unstructured pointcloud instead of
                polyline(s).

        Returns a list of Polylines.
        '''
        from polliwog import Polyline

        # 1: Select those faces that intersect the plane, fs
        fs = self.faces_intersecting_plane(plane)

        if len(fs) == 0:
            # Nothing intersects
            if ret_pointcloud:
                return np.zeros((0, 3))
            elif neighborhood:
                return None
            else:
                return []

        # and edges of those faces
        es = np.vstack((fs[:, (0, 1)], fs[:, (1, 2)], fs[:, (2, 0)]))

        # 2: Find the edges where each of those faces actually cross the plane
        intersection_map = _EdgeMap()

        pts, pt_is_valid = plane.line_segment_xsections(self.v[es[:, 0]], self.v[es[:, 1]])
        valid_pts = pts[pt_is_valid]
        valid_es = es[pt_is_valid]
        for val, e in zip(valid_pts, valid_es):
            if not intersection_map.contains(e[0], e[1]):
                intersection_map.add(e[0], e[1], val)
        verts = np.array(intersection_map.values)

        # 3: Build the edge adjacency graph
        G = _Graph(verts.shape[0])
        for f in fs:
            # Since we're dealing with a triangle that intersects the plane, exactly two of the edges
            # will intersect (note that the only other sorts of "intersections" are one edge in
            # plane or all three edges in plane, which won't be picked up by mesh_intersecting_faces).
            e0 = intersection_map.index(f[0], f[1])
            e1 = intersection_map.index(f[0], f[2])
            e2 = intersection_map.index(f[1], f[2])
            if e0 is None:
                G.add_edge(e1, e2)
            elif e1 is None:
                G.add_edge(e0, e2)
            else:
                G.add_edge(e0, e1)

        # 4: Find the paths for each component
        components = []
        components_closed = []
        while len(G) > 0:
            path = G.pop_euler_path()
            if path is None:
                raise ValueError("mesh slice has too many odd degree edges; can't find a path along the edge")
            component_verts = verts[path]

            if np.all(component_verts[0] == component_verts[-1]):
                # Because the closed polyline will make that last link:
                component_verts = np.delete(component_verts, 0, axis=0)
                components_closed.append(True)
            else:
                components_closed.append(False)
            components.append(component_verts)

        # 6 (optional - only if 'neighborhood' is provided): Use a KDTree to
        # select the component with minimal distance to 'neighborhood'.
        if neighborhood is not None and len(components) > 1:
            from scipy.spatial import cKDTree  # First thought this warning was caused by a pythonpath problem, but it seems more likely that the warning is caused by scipy import hackery. pylint: disable=no-name-in-module

            kdtree = cKDTree(neighborhood)

            # number of components will not be large in practice, so this loop won't hurt
            means = [np.mean(kdtree.query(component)[0]) for component in components]
            index = np.argmin(means)
            if ret_pointcloud:
                return components[index]
            else:
                return Polyline(components[index], is_closed=components_closed[index])
        elif neighborhood is not None and len(components) == 1:
            if ret_pointcloud:
                return components[0]
            else:
                return Polyline(components[0], is_closed=components_closed[0])
        else:
            # No neighborhood provided, so return all the components, either in
            # a pointcloud or as separate polylines.
            if ret_pointcloud:
                return np.vstack(components)
            else:
                return [Polyline(v, is_closed=closed) for v, closed in zip(components, components_closed)]
