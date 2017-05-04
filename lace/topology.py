# pylint: disable=attribute-defined-outside-init, access-member-before-definition, len-as-condition
import zlib

def get_vert_connectivity(mesh):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15, 12), that means vertex 15 is connected
    by an edge to vertex 12."""
    import numpy as np
    import scipy.sparse as sp
    from blmath.numerics.matlab import row

    vpv = sp.csc_matrix((len(mesh.v), len(mesh.v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh.f[:, i]
        JS = mesh.f[:, (i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

_vertices_per_edge_cache = {}
def get_vertices_per_edge(mesh, faces_per_edge=None):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""
    import numpy as np
    import scipy.sparse as sp
    from blmath.numerics.matlab import row, col

    faces = mesh.f
    suffix = str(zlib.crc32(faces_per_edge.flatten())) if faces_per_edge != None else ''
    cache_key = str(zlib.crc32(faces.flatten())) + '_' + suffix
    if cache_key in _vertices_per_edge_cache:
        return _vertices_per_edge_cache[cache_key]
    else:
        if faces_per_edge != None:
            #result = np.asarray([vertices_in_common(face_pair[0], face_pair[1]) for face_pair in mesh.f[faces_per_edge]])
            #new code above is a few times faster
            result = np.asarray(np.vstack([row(np.intersect1d(mesh.f[k[0]], mesh.f[k[1]])) for k in faces_per_edge]), np.uint32)
        else:
            vc = sp.coo_matrix(get_vert_connectivity(mesh))
            result = np.hstack((col(vc.row), col(vc.col)))
            result = result[result[:, 0] < result[:, 1]] # for uniqueness

        _vertices_per_edge_cache[cache_key] = result
        return result

def gen_coupling_weights(sm_template, default_weights=1.0):
    """
    feet and hands have a higher weight, and rest of parts will have default_weights
    TODO: Remove and use `generate_parts_coupling_weights` from
    `bodylabs.util.coupling`
    """
    # from bodylabs.util.stats.robustifiers import Sigmoid
    import numpy as np
    Sigmoid = lambda x, scale: x * scale / np.sqrt(x**2 + scale**2)

    m = sm_template

    min_y = m.v[:, 1].min()
    min_x = m.v[:, 0].min()
    max_x = m.v[:, 0].max()

    def p(w, u=0.5, l=0.3):
        w[w > u] = u
        w[w < l] = l
        w = (w - l) / (u - l)
        return w

    min_hands_w = Sigmoid(x=np.abs(m.v[:, 0]-min_x), scale=0.1)
    min_hands_w = 1.0 - min_hands_w / (min_hands_w.max() - min_hands_w.min())
    min_hands_w = p(min_hands_w)

    max_hands_w = Sigmoid(x=np.abs(m.v[:, 0]-max_x), scale=0.1)
    max_hands_w = 1.0 - max_hands_w / (max_hands_w.max() - max_hands_w.min())
    max_hands_w = p(max_hands_w)

    feet_weights = Sigmoid(x=np.abs(m.v[:, 1]-min_y), scale=0.1)
    feet_weights = 1.0 - feet_weights / (feet_weights.max() - feet_weights.min())
    feet_weights = p(feet_weights, u=0.8, l=0.2)

    weights = np.maximum(min_hands_w, max_hands_w)
    weights = np.maximum(weights, feet_weights)

    weights = (3.0 - default_weights) * weights
    weights = weights + default_weights

    return weights



def quads_to_tris(quads):
    '''
    Convert quad faces to triangular faces.

    quads: An nx4 array.

    Return a 2nx3 array.

    '''
    import numpy as np
    tris = np.empty((2 * len(quads), 3))
    tris[0::2, :] = quads[:, [0, 1, 2]]
    tris[1::2, :] = quads[:, [0, 2, 3]]
    return tris

class MeshMixin(object):
    def faces_by_vertex(self, as_sparse_matrix=False):
        import numpy as np
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
        import numpy as np
        included_vertices = np.zeros(self.v.shape[0], dtype=bool)
        included_vertices[np.array(v_indices, dtype=np.uint32)] = True
        faces_with_verts = included_vertices[self.f].all(axis=1)
        if as_boolean:
            return faces_with_verts
        return np.nonzero(faces_with_verts)[0]

    def transfer_segm(self, mesh, exclude_empty_parts=True):
        import numpy as np
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
        import numpy as np
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
        import numpy as np
        joint_locations = {}
        for name in self.joint_names:
            joint_locations[name] = self.joint_regressors[name]['offset'] + np.sum(self.v[self.joint_regressors[name]['v_indices']].T*self.joint_regressors[name]['coeff'], axis=1)
        return joint_locations

    # creates joint_regressors from a list of joint names and a per joint list of vertex indices (e.g. a ring of vertices)
    # For the regression coefficients, all vertices for a given joint are given equal weight
    def set_joints(self, joint_names, vertex_indices):
        import numpy as np
        self.joint_regressors = {}
        for name, indices in zip(joint_names, vertex_indices):
            self.joint_regressors[name] = {'v_indices': indices, 'coeff': [1.0/len(indices)]*len(indices), 'offset': np.array([0., 0., 0.])}

    def uniquified_mesh(self):
        """This function returns a copy of the mesh in which vertices are copied such that
        each vertex appears in only one face, and hence has only one texture"""
        import numpy as np
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
        import numpy as np

        if self.v is None:
            return

        initial_num_verts = self.v.shape[0]
        if self.f is not None:
            initial_num_faces = self.f.shape[0]
            f_indices_to_keep = self.all_faces_with_verts(indices_to_keep, as_boolean=True)

        # Why do we test this? Don't know. But we do need to test it before we
        # mutate self.v.
        vn_should_update = self.vn is not None and self.vn.shape[0] == initial_num_verts
        vc_should_update = self.vc is not None and self.vc.shape[0] == initial_num_verts

        self.v = self.v[np.array(indices_to_keep, dtype=np.uint32)]

        if vn_should_update:
            self.vn = self.vn[indices_to_keep]
        if vc_should_update:
            self.vc = self.vc[indices_to_keep]

        if self.f is not None:
            v_old_to_new = np.zeros(initial_num_verts, dtype=int)
            f_old_to_new = np.zeros(initial_num_faces, dtype=int)

            v_old_to_new[np.array(indices_to_keep, dtype=np.uint32)] = np.arange(len(indices_to_keep), dtype=int)
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
        import numpy as np
        return self.keep_vertices(np.setdiff1d(np.arange(self.v.shape[0]), v_list))

    def point_cloud(self):
        return self.__class__(v=self.v, f=[], vc=self.vc) if self.vc is not None else self.__class__(v=self.v, f=[])

    def remove_faces(self, face_indices_to_remove):
        import numpy as np
        self.f = np.delete(self.f, face_indices_to_remove, 0)
        return self

    def flip_faces(self):
        self.f = self.f.copy()
        for i in range(len(self.f)):
            self.f[i] = self.f[i][::-1]
        if self.ft is not None:
            for i in range(len(self.f)):
                self.ft[i] = self.ft[i][::-1]
        return self

    def subdivide_triangles(self):
        import numpy as np
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
        import numpy as np
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
        import numpy as np
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
        import numpy as np
        if not lines:
            return

        v_lines = np.vstack([l.v for l in lines])
        v_index_offset = np.cumsum([0] + [len(l.v) for l in lines])
        e_lines = np.vstack([l.e + v_index_offset[i] for i, l in enumerate(lines)])
        num_body_verts = self.v.shape[0]
        self.v = np.vstack([self.v, v_lines])
        self.e = e_lines + num_body_verts
