# pylint: disable=attribute-defined-outside-init, access-member-before-definition, len-as-condition
from cached_property import cached_property
import numpy as np


def quads_to_tris(quads, ret_mapping=False):
    '''
    Convert quad faces to triangular faces.

    quads: An nx4 array.
    ret_mapping: A bool.

    When `ret_mapping` is `True`, return a 2nx3 array of new triangles and a 2nx3
    array mapping old quad indices to new trangle indices.

    When `ret_mapping` is `False`, return the 2nx3 array of triangles.
    '''
    tris = np.empty((2 * len(quads), 3))
    tris[0::2, :] = quads[:, [0, 1, 2]]
    tris[1::2, :] = quads[:, [0, 2, 3]]
    if ret_mapping:
        f_old_to_new = np.arange(len(tris)).reshape(-1, 2)
        return tris, f_old_to_new
    else:
        return tris


def vertices_in_common(face_1, face_2):
    "returns the two vertices shared by two faces"
    # So... on a 10000 iteration timing run, np.intersect1d takes 0.2s, the very
    # complicated code we pulled from core takes 0.05s, and this runs in 0.007s.
    # Just goes to show that sometimes simplest is best... The easy timing script
    # for things like this is:
    # import timeit; print timeit.timeit('vertices_in_common([0, 1, 2], [0, 1, 3])', setup='from lace.topology import vertices_in_common', number=10000)
    return [x for x in face_1 if x in face_2]


class MeshMixin(object):
    def faces_by_vertex(self, as_sparse_matrix=False):
        import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = self.f.flatten()
            col = np.array([range(self.f.shape[0])]*3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(self.v.shape[0], self.f.shape[0]))
        return faces_by_vertex

    def all_faces_with_verts(self, v_indices, as_boolean=False):
        '''
        returns all of the faces that contain at least one of the vertices in v_indices
        '''
        included_vertices = np.zeros(self.v.shape[0], dtype=bool)
        included_vertices[np.array(v_indices, dtype=np.uint32)] = True
        faces_with_verts = included_vertices[self.f].all(axis=1)
        if as_boolean:
            return faces_with_verts
        return np.nonzero(faces_with_verts)[0]

    def transfer_segm(self, mesh, exclude_empty_parts=True):
        self.segm = {}
        if mesh.segm is not None:
            face_centers = np.array([self.v[face, :].mean(axis=0) for face in self.f])
            closest_faces, _ = mesh.closest_faces_and_points(face_centers)
            mesh_parts_by_face = mesh.parts_by_face()
            parts_by_face = [mesh_parts_by_face[face] for face in closest_faces.flatten()]
            self.segm = dict([(part, []) for part in mesh.segm.keys()])
            for face, part in enumerate(parts_by_face):
                if part:
                    self.segm[part].append(face)
            for part in self.segm.keys():
                self.segm[part].sort()
                if exclude_empty_parts and not self.segm[part]:
                    del self.segm[part]

    def vertex_indices_in_segments(self, segments, ret_face_indices=False):
        '''
        Given a list of segment names, return an array of vertex indices for
        all the vertices in those faces.

        Args:
            segments: a list of segment names,
            ret_face_indices: if it is `True`, returns face indices
        '''
        import warnings

        face_indices = np.array([])
        vertex_indices = np.array([])
        if self.segm is not None:
            try:
                segments = [self.segm[name] for name in segments]
            except KeyError as e:
                raise ValueError('Unknown segments {}. Consier using Mesh.clean_segments on segments'.format(e.args[0]))
            face_indices = np.unique(np.concatenate(segments))
            vertex_indices = np.unique(np.ravel(self.f[face_indices]))
        else:
            warnings.warn('self.segm is None, will return empty array')

        if ret_face_indices:
            return vertex_indices, face_indices
        else:
            return vertex_indices

    def clean_segments(self, segments):
        """Return a list of segments that are in self.segm
        """
        return [name for name in segments if name in self.segm]

    def keep_segments(self, segments_to_keep, preserve_segmentation=True):
        '''
        Keep the faces and vertices for given segments, discarding all others.
        When preserve_segmentation is false self.segm is discarded for speed.
        '''
        v_ind, f_ind = self.vertex_indices_in_segments(segments_to_keep, ret_face_indices=True)
        self.segm = {name: self.segm[name] for name in segments_to_keep}

        if not preserve_segmentation:
            self.segm = None
            self.f = self.f[f_ind]
            if self.ft is not None:
                self.ft = self.ft[f_ind]

        self.keep_vertices(v_ind)

    def remove_segments(self, segments_to_remove):
        ''' Remove the faces and vertices for given segments, keeping all others.

        Args:
            segments_to_remove: a list of segnments whose vertices will be removed
        '''
        v_ind = self.vertex_indices_in_segments(segments_to_remove)
        self.segm = {name: faces for name, faces in self.segm.iteritems() if name not in segments_to_remove}
        self.remove_vertices(v_ind)

    @property
    def verts_by_segm(self):
        return dict((segment, sorted(set(self.f[indices].flatten()))) for segment, indices in self.segm.items())

    def parts_by_face(self):
        segments_by_face = ['']*len(self.f)
        for part in self.segm.keys():
            for face in self.segm[part]:
                segments_by_face[face] = part
        return segments_by_face

    def verts_in_common(self, segments):
        """
        returns array of all vertex indices common to each segment in segments"""
        verts_by_segm = self.verts_by_segm
        return sorted(reduce(lambda s0, s1: s0.intersection(s1),
                             [set(verts_by_segm[segm]) for segm in segments]))
        ## indices of vertices in the faces of the first segment
        #indices = self.verts_by_segm[segments[0]]
        #for segment in segments[1:]:
        #    indices = sorted([index for index in self.verts_by_segm[segment] if index in indices]) # Intersect current segment with current indices
        #return sorted(set(indices))

    @property
    def joint_names(self):
        return self.joint_regressors.keys()

    @property
    def joint_xyz(self):
        joint_locations = {}
        for name in self.joint_names:
            joint_locations[name] = self.joint_regressors[name]['offset'] + np.sum(self.v[self.joint_regressors[name]['v_indices']].T*self.joint_regressors[name]['coeff'], axis=1)
        return joint_locations

    # creates joint_regressors from a list of joint names and a per joint list of vertex indices (e.g. a ring of vertices)
    # For the regression coefficients, all vertices for a given joint are given equal weight
    def set_joints(self, joint_names, vertex_indices):
        self.joint_regressors = {}
        for name, indices in zip(joint_names, vertex_indices):
            self.joint_regressors[name] = {'v_indices': indices, 'coeff': [1.0/len(indices)]*len(indices), 'offset': np.array([0., 0., 0.])}

    def uniquified_mesh(self):
        """This function returns a copy of the mesh in which vertices are copied such that
        each vertex appears in only one face, and hence has only one texture"""
        from lace.mesh import Mesh
        new_mesh = Mesh(v=self.v[self.f.flatten()], f=np.array(range(len(self.f.flatten()))).reshape(-1, 3))

        if self.vn is None:
            self.reset_normals()
        new_mesh.vn = self.vn[self.f.flatten()]

        if self.vt is not None:
            new_mesh.vt = self.vt[self.ft.flatten()]
            new_mesh.ft = new_mesh.f.copy()
        return new_mesh

    def downsampled_mesh(self, step):
        """Returns a downsampled copy of this mesh.

        Args:
            step: the step size for the sampling

        Returns:
            a new, downsampled Mesh object.

        Raises:
            ValueError if this Mesh has faces.
        """
        from lace.mesh import Mesh

        if self.f is not None:
            raise ValueError(
                'Function `downsampled_mesh` does not support faces.')

        low = Mesh()
        if self.v is not None:
            low.v = self.v[::step]
        if self.vc is not None:
            low.vc = self.vc[::step]
        return low

    def keep_vertices(self, indices_to_keep, ret_kept_faces=False):
        '''
        Keep the given vertices and discard the others, and any faces to which
        they may belong.


        If `ret_kept_faces` is `True`, return the original indices of the kept
        faces. Otherwise return `self` for chaining.

        '''
        if self.v is None:
            return

        indices_to_keep = np.array(indices_to_keep, dtype=np.uint32)

        initial_num_verts = self.v.shape[0]
        if self.f is not None:
            initial_num_faces = self.f.shape[0]
            f_indices_to_keep = self.all_faces_with_verts(indices_to_keep, as_boolean=True)

        # Why do we test this? Don't know. But we do need to test it before we
        # mutate self.v.
        vn_should_update = self.vn is not None and self.vn.shape[0] == initial_num_verts
        vc_should_update = self.vc is not None and self.vc.shape[0] == initial_num_verts

        self.v = self.v[indices_to_keep]

        if vn_should_update:
            self.vn = self.vn[indices_to_keep]
        if vc_should_update:
            self.vc = self.vc[indices_to_keep]

        if self.f is not None:
            v_old_to_new = np.zeros(initial_num_verts, dtype=int)
            f_old_to_new = np.zeros(initial_num_faces, dtype=int)

            v_old_to_new[indices_to_keep] = np.arange(len(indices_to_keep), dtype=int)
            self.f = v_old_to_new[self.f[f_indices_to_keep]]
            f_old_to_new[f_indices_to_keep] = np.arange(self.f.shape[0], dtype=int)

        else:
            # Make the code below work, in case there is somehow degenerate
            # segm even though there are no faces.
            f_indices_to_keep = []

        if self.segm is not None:
            new_segm = {}
            for segm_name, segm_faces in self.segm.items():
                faces = np.array(segm_faces, dtype=int)
                valid_faces = faces[f_indices_to_keep[faces]]
                if len(valid_faces):
                    new_segm[segm_name] = f_old_to_new[valid_faces]
            self.segm = new_segm if new_segm else None

        if hasattr(self, '_raw_landmarks') and self._raw_landmarks is not None:
            self.recompute_landmarks()

        return np.nonzero(f_indices_to_keep)[0] if ret_kept_faces else self

    def remove_vertices(self, v_list):
        return self.keep_vertices(np.setdiff1d(np.arange(self.v.shape[0]), v_list))

    def point_cloud(self):
        return self.__class__(v=self.v, f=[], vc=self.vc) if self.vc is not None else self.__class__(v=self.v, f=[])

    def remove_faces(self, face_indices_to_remove):
        self.f = np.delete(self.f, face_indices_to_remove, 0)
        return self

    def flip_faces(self, face_indices_to_flip=()):
        self.f[face_indices_to_flip] = np.fliplr(self.f[face_indices_to_flip])
        if self.ft is not None:
            self.ft[face_indices_to_flip] = np.fliplr(self.ft[face_indices_to_flip])
        return self

    def subdivide_triangles(self):
        new_faces = []
        new_vertices = self.v.copy()
        for face in self.f:
            face_vertices = np.array([self.v[face[0], :], self.v[face[1], :], self.v[face[2], :]])
            new_vertex = np.mean(face_vertices, axis=0)
            new_vertices = np.vstack([new_vertices, new_vertex])
            new_vertex_index = len(new_vertices) - 1
            if len(new_faces):
                new_faces = np.vstack([new_faces, [face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
            else:
                new_faces = np.array([[face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
        self.v = new_vertices
        self.f = new_faces

        if self.vt is not None:
            new_ft = []
            new_texture_coordinates = self.vt.copy()
            for face_texture in self.ft:
                face_texture_coordinates = np.array([self.vt[face_texture[0], :], self.vt[face_texture[1], :], self.vt[face_texture[2], :]])
                new_texture_coordinate = np.mean(face_texture_coordinates, axis=0)
                new_texture_coordinates = np.vstack([new_texture_coordinates, new_texture_coordinate])
                new_texture_index = len(new_texture_coordinates) - 1
                if len(new_ft):
                    new_ft = np.vstack([new_ft, [face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
                else:
                    new_ft = np.array([[face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
            self.vt = new_texture_coordinates
            self.ft = new_ft
        return self

    def concatenate_mesh(self, mesh):
        if len(self.v) == 0:
            self.f = mesh.f.copy()
            self.v = mesh.v.copy()
            self.vc = mesh.vc.copy() if mesh.vc is not None else None
        elif len(mesh.v):
            self.f = np.concatenate([self.f, mesh.f.copy() + len(self.v)])
            self.v = np.concatenate([self.v, mesh.v])
            self.vc = np.concatenate([self.vc, mesh.vc]) if (mesh.vc is not None and self.vc is not None) else None
        return self

    # new_ordering specifies the new index of each vertex. If new_ordering[i] = j,
    # vertex i should now be the j^th vertex. As such, each entry in new_ordering should be unique.
    def reorder_vertices(self, new_ordering, new_normal_ordering=None):
        if new_normal_ordering is None:
            new_normal_ordering = new_ordering
        inverse_ordering = np.zeros(len(new_ordering), dtype=int)
        for i, j in enumerate(new_ordering):
            inverse_ordering[j] = i
        inverse_normal_ordering = np.zeros(len(new_normal_ordering), dtype=int)
        for i, j in enumerate(new_normal_ordering):
            inverse_normal_ordering[j] = i
        self.v = self.v[inverse_ordering]
        if self.vn is not None:
            self.vn = self.vn[inverse_normal_ordering]
        for i in range(len(self.f)):
            self.f[i] = np.array([new_ordering[vertex_index] for vertex_index in self.f[i]])
            if self.fn is not None:
                self.fn[i] = np.array([new_normal_ordering[normal_index] for normal_index in self.fn[i]])

    @classmethod
    def create_from_mesh_and_lines(cls, mesh, lines):
        '''
        Return a copy of mesh with line vertices and edges added.

        mesh: A Mesh
        lines: A list of Polyline or Lines objects.

        '''
        mesh_with_lines = mesh.copy()
        mesh_with_lines.add_lines(lines)
        return mesh_with_lines

    def add_lines(self, lines):
        '''
        Add line vertices and edges to the mesh.

        lines: A list of Polyline or Lines objects.

        '''
        if not lines:
            return

        v_lines = np.vstack([l.v for l in lines])
        v_index_offset = np.cumsum([0] + [len(l.v) for l in lines])
        e_lines = np.vstack([l.e + v_index_offset[i] for i, l in enumerate(lines)])
        num_body_verts = self.v.shape[0]
        self.v = np.vstack([self.v, v_lines])
        self.e = e_lines + num_body_verts

    @property
    def vert_connectivity(self):
        """Returns a sparse matrix (of size #verts x #verts) where each nonzero
        element indicates a neighborhood relation. For example, if there is a
        nonzero element in position (15, 12), that means vertex 15 is connected
        by an edge to vertex 12."""
        import scipy.sparse as sp
        from blmath.numerics.matlab import row
        vpv = sp.csc_matrix((len(self.v), len(self.v)))
        # for each column in the faces...
        for i in range(3):
            IS = self.f[:, i]
            JS = self.f[:, (i+1)%3]
            data = np.ones(len(IS))
            ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
            mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
            vpv = vpv + mtx + mtx.T
        return vpv

    @property
    def vert_opposites_per_edge(self):
        """Returns a dictionary from vertidx-pairs to opposites.
        For example, a key consist of [4, 5)] meaning the edge between
        vertices 4 and 5, and a value might be [10, 11] which are the indices
        of the vertices opposing this edge."""
        result = {}
        for f in self.f:
            for i in range(3):
                key = [f[i], f[(i+1)%3]]
                key.sort()
                key = tuple(key)
                val = f[(i+2)%3]

                if key in result:
                    result[key].append(val)
                else:
                    result[key] = [val]
        return result

    @cached_property
    def faces_per_edge(self):
        """Returns an Ex2 array of adjacencies between faces, where
        each element in the array is a face index. Each edge is included
        only once. Edges that are not shared by 2 faces are not included."""
        import scipy.sparse as sp
        from blmath.numerics.matlab import col
        IS = np.repeat(np.arange(len(self.f)), 3)
        JS = self.f.ravel()
        data = np.ones(IS.size)
        last_referenced_vert = int(np.max(self.f))
        f2v = sp.csc_matrix((data, (IS, JS)), shape=(len(self.f), last_referenced_vert + 1))
        f2f = f2v.dot(f2v.T)
        f2f = f2f.tocoo()
        f2f = np.hstack((col(f2f.row), col(f2f.col), col(f2f.data)))
        which = (f2f[:, 0] < f2f[:, 1]) & (f2f[:, 2] >= 2)
        return np.asarray(f2f[which, :2], np.uint32)

    @cached_property
    def vertices_per_edge(self):
        """Returns an Ex2 array of adjacencies between vertices, where
        each element in the array is a vertex index. Each edge is included
        only once. Edges that are not shared by 2 faces are not included."""
        return np.asarray([vertices_in_common(e[0], e[1]) for e in self.f[self.faces_per_edge]])

    def get_vertices_to_edges_matrix(self, want_xyz=True):
        """Returns a matrix M, which if multiplied by vertices,
        gives back edges (so "e = M.dot(v)"). Note that this generates
        one edge per edge, *not* two edges per triangle.

        Args:
            want_xyz: if true, takes and returns xyz coordinates, otherwise
                takes and returns x *or* y *or* z coordinates
        """
        import scipy.sparse as sp

        vpe = np.asarray(self.vertices_per_edge, dtype=np.int32)
        IS = np.repeat(np.arange(len(vpe)), 2)
        JS = vpe.flatten()
        data = np.ones_like(vpe)
        data[:, 1] = -1
        data = data.flatten()

        if want_xyz:
            IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
            JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
            data = np.concatenate((data, data, data))

        ij = np.vstack((IS.flatten(), JS.flatten()))
        return sp.csc_matrix((data, ij))

    @cached_property
    def vertices_to_edges_matrix(self):
        return self.get_vertices_to_edges_matrix(want_xyz=True)

    @cached_property
    def vertices_to_edges_matrix_single_axis(self):
        return self.get_vertices_to_edges_matrix(want_xyz=False)

    def remove_redundant_verts(self, eps=1e-10):
        """Given verts and faces, this remove colocated vertices"""
        from scipy.spatial import cKDTree # FIXME pylint: disable=no-name-in-module
        fshape = self.f.shape
        tree = cKDTree(self.v)
        close_pairs = list(tree.query_pairs(eps))
        if close_pairs:
            close_pairs = np.sort(close_pairs, axis=1)
            # update faces to not refer to redundant vertices
            equivalent_verts = np.arange(self.v.shape[0])
            for v1, v2 in close_pairs:
                if equivalent_verts[v2] > v1:
                    equivalent_verts[v2] = v1
            self.f = equivalent_verts[self.f.flatten()].reshape((-1, 3))
            # get rid of unused verts, and update faces accordingly
            vertidxs_left = np.unique(self.f)
            repl = np.arange(np.max(self.f)+1)
            repl[vertidxs_left] = np.arange(len(vertidxs_left))
            self.v = self.v[vertidxs_left]
            self.f = repl[self.f].reshape((-1, fshape[1]))

    def remove_unreferenced_vertices(self):
        self.keep_vertices(np.unique(self.f.reshape(-1)))

    def has_same_topology(self, other_mesh):
        return self.has_same_len_attr(other_mesh, 'v') and \
            self.has_equal_attr(other_mesh, 'f')
