# pylint: disable=attribute-defined-outside-init
import numpy as np
import vg

def reorient_faces_using_normals(mesh):
    """
    Using face normals, infer counterclockwise winding and canonicalize
    the faces.

    Return a list of indices of faces which were flipped.
    """
    import math
    from polliwog.tri.surface_normals import surface_normal
    if mesh.fn is None:
        raise ValueError("Face normals are required")
    normals_from_winding = surface_normal(mesh.v[mesh.f])
    deviation_angle = vg.angle(mesh.fn, normals_from_winding, units="rad")
    need_flipping, = np.nonzero(deviation_angle > 0.5 * math.pi)
    mesh.flip_faces(need_flipping)
    return need_flipping

class MeshMixin(object):
    def estimate_vertex_normals(self, face_to_verts_sparse_matrix=None):
        from blmath.optimization.objectives.normals import TriNormalsScaled
        face_normals = TriNormalsScaled(self.v, self.f).r.reshape(-1, 3)
        ftov = face_to_verts_sparse_matrix if face_to_verts_sparse_matrix else self.faces_by_vertex(as_sparse_matrix=True)
        non_scaled_normals = ftov*face_normals
        norms = (np.sum(non_scaled_normals**2.0, axis=1)**0.5).T
        norms[norms == 0] = 1.0
        self.vn = (non_scaled_normals.T/norms).T
        return self.vn # for backwards compatibility

    def barycentric_coordinates_for_points(self, points, face_indices):
        from polliwog.tri.barycentric import barycentric_coordinates_of_projection
        vertex_indices = self.f[face_indices]
        vertices = self.v[vertex_indices]
        coeffs = barycentric_coordinates_of_projection(
            points,
            vertices[:, 0],
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0])
        return vertex_indices, coeffs

    def reset_normals(self, face_to_verts_sparse_matrix=None):
        self.vn = self.estimate_vertex_normals(face_to_verts_sparse_matrix)

    def reset_face_normals(self):
        if not hasattr(self, 'vn'):
            self.reset_normals()
        self.fn = self.f

    def scale(self, scale_factor):
        if self.v is not None:
            self.v *= scale_factor

    def convert_units(self, from_units, to_units):
        '''
        Convert the mesh from one set of units to another.

        These calls are equivalent:

        - mesh.convert_units(from_units='cm', to_units='m')
        - mesh.scale(.01)

        '''
        from blmath import units
        factor = units.factor(
            from_units=from_units,
            to_units=to_units,
            units_class='length'
        )
        self.scale(factor)

    def predict_body_units(self):
        '''
        There is no prediction for united states unit system.
        This may fail when a mesh is not axis-aligned
        '''
        longest_dist = np.max(np.max(self.v, axis=0) - np.min(self.v, axis=0))
        if round(longest_dist / 1000) > 0:
            return 'mm'
        if round(longest_dist / 100) > 0:
            return 'cm'
        if round(longest_dist / 10) > 0:
            return 'dm'
        return 'm'

    def assert_body_units(self, expected):
        if not self.predict_body_units() == expected:
            raise ValueError('Expected body to be in {} but it appears to be in {}')

    def rotate(self, rotation_matrix):
        if np.array(rotation_matrix).shape != (3, 3):
            import cv2
            rotation_matrix = cv2.Rodrigues(np.array(rotation_matrix))[0]
        self.v = np.dot(self.v, rotation_matrix.T)

    def translate(self, translation):
        if self.v is not None:
            self.v += translation

    def reorient(self, up, look):
        '''
        Reorient the mesh by specifying two vectors.

        up: The foot-to-head direction.
        look: The direction the body is facing.

        In the result, the up will end up along +y, and look along +z
        (i.e. facing towards a default OpenGL camera).

        '''
        from polliwog.transform.rotation import rotation_from_up_and_look
        from blmath.numerics import as_numeric_array

        up = as_numeric_array(up, (3,))
        look = as_numeric_array(look, (3,))

        if self.v is not None:
            self.v = np.dot(rotation_from_up_and_look(up, look), self.v.T).T

    def flip(self, axis=0, preserve_centroid=False):
        '''
        Flip the mesh across the given axis: 0 for x, 1 for y, 2 for z.

        When `preserve_centroid` is True, translate after flipping to
        preserve the location of the centroid.

        '''
        self.v[:, axis] *= -1

        if preserve_centroid:
            self.v[:, axis] -= 2 * self.centroid[0]

        self.flip_faces()

    # TODO : many of the methods below (e.g. centroid, floor_point, bounding_box)
    # are highly sensative to outliers. A nice compliment to these methods would
    # be a way to copy/modify the mesh so some percentile of outlier are removed.
    @property
    def centroid(self):
        '''
        Return the geometric center.

        '''
        if self.v is None:
            raise ValueError('Mesh has no vertices; centroid is not defined')

        return np.mean(self.v, axis=0)

    def recenter_on_centroid(self):
        self.translate(-self.centroid)

    @property
    def floor_point(self):
        '''
        Return the point on the floor that lies below the centroid.

        '''
        floor_point = self.centroid
        # y to floor
        floor_point[1] = self.v[:, 1].min()
        return floor_point

    @property
    def floor_plane(self):
        from polliwog import Plane
        return Plane(self.floor_point, vg.basis.y)

    def recenter_over_floor(self):
        self.translate(-self.floor_point)

    @property
    def bounding_box(self):
        from polliwog import Box

        if self.v is None:
            raise ValueError('Mesh has no vertices; bounding box is not defined')

        return Box.from_points(self.v)

    def almost_on_floor(self, atol=1e-08):
        if self.v is None:
            raise ValueError('Mesh has no vertices; floor is not defined')

        min_y = np.min(self.v, axis=0)[1]
        return np.isclose(min_y, 0., rtol=0, atol=atol)

    def apex(self, axis):
        '''
        Find the most extreme vertex in the direction of the axis provided.

        axis: A vector, which is an 3x1 np.array.

        '''
        return vg.apex(self.v, axis)

    def first_blip(self, squash_axis, origin, initial_direction):
        '''
        Flatten the mesh to the plane, dropping the dimension identified by
        `squash_axis`: 0 for x, 1 for y, 2 for z. Cast a ray from `origin`,
        pointing along `initial_direction`. Sweep the ray, like a radar, until
        encountering the mesh, and return that vertex: like the first blip of
        the radar. The radar works as if on a point cloud: it sees sees only
        vertices, not edges.

        The ray sweeps clockwise and counterclockwise at the same time, and
        returns the first point it hits.

        If no intersection occurs within 90 degrees, return None.

        `initial_direction` need not be normalized.

        '''
        from blmath.numerics import as_numeric_array

        origin = vg.reject_axis(as_numeric_array(origin, (3,)), axis=squash_axis, squash=True)
        initial_direction = vg.reject_axis(as_numeric_array(initial_direction, (3,)), axis=squash_axis, squash=True)
        vertices = vg.reject_axis(self.v, axis=squash_axis, squash=True)

        origin_to_mesh = vg.normalize(vertices - origin)
        cosines = vg.normalize(initial_direction).dot(origin_to_mesh.T).T
        index_of_first_blip = np.argmax(cosines)

        return self.v[index_of_first_blip]

    def cut_across_axis(self, dim, minval=None, maxval=None):
        '''
        Cut the mesh by a plane, discarding vertices that lie behind that
        plane. Or cut the mesh by two parallel planes, discarding vertices
        that lie outside them.

        The region to keep is defined by an axis of perpendicularity,
        specified by `dim`: 0 means x, 1 means y, 2 means z. `minval`
        and `maxval` indicate the portion of that axis to keep.

        Return the original indices of the kept vertices.

        '''
        # vertex_mask keeps track of the vertices we want to keep.
        vertex_mask = np.ones((len(self.v),), dtype=bool)

        if minval is not None:
            predicate = self.v[:, dim] >= minval
            vertex_mask = np.logical_and(vertex_mask, predicate)

        if maxval is not None:
            predicate = self.v[:, dim] <= maxval
            vertex_mask = np.logical_and(vertex_mask, predicate)

        vertex_indices = np.flatnonzero(vertex_mask)
        self.keep_vertices(vertex_indices)

        return vertex_indices

    def cut_across_axis_by_percentile(self, dim, minpct=0, maxpct=100):
        '''
        Like cut_across_axis, except the subset of vertices is
        constrained by percentile of the data along a given axis
        instead of specific values. (See numpy.percentile.)

        For example, if the mesh has 50,000 vertices, `dim` is 2, and
        `minpct` is 10, this method drops the 5,000 vertices which are
        furthest along the +z axis.

        See numpy.percentile

        Return the original indices of the kept vertices.

        '''
        value_range = np.percentile(self.v[:, dim], (minpct, maxpct))
        return self.cut_across_axis(dim, *value_range)

    def cut_by_plane(self, plane, inverted=False):
        '''
        Like cut_across_axis, but works with an arbitrary plane. Keeps
        vertices that lie in front of the plane (i.e. in the direction
        of the plane normal).

        inverted: When `True`, invert the logic, to keep the vertices
          that lie behind the plane instead.

        Return the original indices of the kept vertices.

        '''
        vertices_to_keep = plane.points_in_front(self.v, inverted=inverted, ret_indices=True)

        self.keep_vertices(vertices_to_keep)

        return vertices_to_keep

    @property
    def surface_area(self):
        '''
        returns the surface area of the mesh
        '''
        return self.surface_areas.sum()

    @property
    def surface_areas(self):
        '''
        returns the surface area of each face
        '''
        e_1 = self.v[self.f[:, 1]] - self.v[self.f[:, 0]]
        e_2 = self.v[self.f[:, 2]] - self.v[self.f[:, 0]]

        cross_products = np.array([e_1[:, 1]*e_2[:, 2] - e_1[:, 2]*e_2[:, 1],
                                   e_1[:, 2]*e_2[:, 0] - e_1[:, 0]*e_2[:, 2],
                                   e_1[:, 0]*e_2[:, 1] - e_1[:, 1]*e_2[:, 0]]).T

        return (0.5)*((cross_products**2.).sum(axis=1)**0.5)

    def reorient_faces_using_normals(self):
        """
        Using face normals, infer counterclockwise winding and canonicalize
        the faces.

        Return a list of indices of faces which were flipped.
        """
        return reorient_faces_using_normals(self)
