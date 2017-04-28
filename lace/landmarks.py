# pylint: disable=attribute-defined-outside-init, len-as-condition, unused-argument

def load_landmarks(filename):
    import re
    from baiji import s3
    from baiji.serialization import json, pickle, yaml
    from lace.serialization import lmrk, meshlab_pickedpoints

    if not s3.exists(filename):
        raise ValueError("Landmark file %s not found" % filename)
    if re.search(".ya{0,1}ml$", filename):
        return yaml.load(filename)
    elif re.search(".json$", filename):
        return json.load(filename)
    elif re.search(".pkl$", filename):
        return pickle.load(filename)
    elif re.search(".lmrk$", filename):
        return lmrk.load(filename)
    elif re.search(".pp$", filename):
        return meshlab_pickedpoints.load(filename)
    else:
        raise ValueError("Landmark file %s is of unknown format" % filename)

class MeshMixin(object):
    '''
    landm: a dictionary of indices
    landm_xyz: landmark locations; these are basically v[landm]
    landm_regressors: regressors; replaces landm, predicts landm_xyz
    '''

    @property
    def landm_xyz(self, ordering=None):
        if ordering is None:
            ordering = self.landm_names

        if len(ordering) == 0: # there are no landmarks
            return {}

        landmark_vertex_locations = (self.landm_xyz_linear_transform(ordering) * self.v.flatten()).reshape(-1, 3)
        return dict(zip(ordering, landmark_vertex_locations))

    def landm_xyz_linear_transform(self, ordering=None):
        import numpy as np
        from blmath.numerics.matlab import col, sparse
        if ordering is None:
            ordering = self.landm_names
        # construct a sparse matrix that converts between the landmark pts and all vertices, with height (# landmarks * 3) and width (# vertices * 3)
        if self.landm_regressors is not None:
            landmark_coefficients = np.hstack([self.landm_regressors[name][1] for name in ordering])
            landmark_indices = np.hstack([self.landm_regressors[name][0] for name in ordering])
            column_indices = np.hstack([col(3*landmark_indices + i) for i in range(3)]).flatten()
            row_indices = np.hstack([[3*index, 3*index + 1, 3*index + 2]*len(self.landm_regressors[ordering[index]][0]) for index in np.arange(len(ordering))])
            values = np.hstack([col(landmark_coefficients) for i in range(3)]).flatten()
            return sparse(row_indices, column_indices, values, 3*len(ordering), 3*self.v.shape[0])
        elif hasattr(self, 'landm') and len(self.landm) > 0:
            landmark_indices = np.array([self.landm[name] for name in ordering])
            column_indices = np.hstack(([col(3*landmark_indices + i) for i in range(3)])).flatten()
            row_indices = np.arange(3*len(ordering))
            return sparse(row_indices, column_indices, np.ones(len(column_indices)), 3*len(ordering), 3*self.v.shape[0])
        else:
            return np.zeros((0, 0))

    @landm_xyz.setter
    def landm_xyz(self, val):
        '''
        Since landm can handle all possibilities, we just delegate to that
        '''
        self.landm = val

    @property
    def landm_regressors(self):
        if hasattr(self, '_landm_regressors'):
            return self._landm_regressors
        else:
            return None

    @landm_regressors.setter
    def landm_regressors(self, val):
        if val is None or len(val) == 0:
            self._landm_regressors = None
        else:
            self._landm_regressors = val
            self.landm = None

    @property
    def landm(self):
        if hasattr(self, '_landm'):
            return self._landm
        else:
            return None
    @landm.setter
    def landm(self, val):
        '''
        Sets landmarks given any of:
         - ppfile
         - ldmk file
         - dict of {name:inds} (i.e. mesh.landm)
         - dict of {name:xyz} (i.e. mesh.landm_xyz)
         - Nx1 array or list of ints (treated as landm, given sequential integers as names)
         - Nx3 array or list of floats (treated as landm_xyz, given sequential integers as names)
         - pkl, json, yaml file containing either of the above dicts or arrays
        '''
        import numpy as np

        if val is None:
            self._landm = None
            self._raw_landmarks = None
        elif isinstance(val, basestring):
            self.landm = load_landmarks(val)
        else:
            if not hasattr(val, 'keys'):
                val = {str(ii): v for ii, v in enumerate(val)}
            landm = {}
            landm_xyz = {}
            filtered_landmarks = []
            for k, v in val.iteritems():
                if isinstance(v, (int, long)):
                    landm[k] = v
                elif len(v) == 3:
                    if np.all(v == [0.0, 0.0, 0.0]):
                        filtered_landmarks.append(k)
                    landm_xyz[k] = v
                else:
                    raise Exception("Can't parse landmark %s: %s" % (k, v))
            if len(filtered_landmarks) > 0:
                import warnings
                warnings.warn("WARNING: the following landmarks are positioned at (0.0, 0.0, 0.0) and were ignored: %s" % ", ".join(filtered_landmarks))
            # We preserve these and calculate everything seperately so that we can recompute_landmarks if v changes
            self._raw_landmarks = {
                'landm': landm,
                'landm_xyz': landm_xyz
            }
            self.recompute_landmarks()

    def recompute_landmarks(self):
        import numpy as np
        landm = self._raw_landmarks['landm']
        landm_xyz = self._raw_landmarks['landm_xyz']
        self._landm = dict(landm.items() + self.compute_landmark_indices(landm_xyz).items())
        if len(self._landm) == 0:
            self._landm = None
            self._landm_regressors = None
        # compute default regressors from landmarks
        elif self.f is not None and len(self.f):
            landmark_points = np.vstack((self.v[landm.values()].reshape(-1, 3), np.array(landm_xyz.values()).reshape(-1, 3)))
            face_indices, closest_points = self.closest_faces_and_points(landmark_points)
            vertex_indices, coefficients = self.barycentric_coordinates_for_points(closest_points, face_indices)
            self._landm_regressors = {name: (vertex_indices[i], coefficients[i]) for i, name in enumerate(landm.keys() + landm_xyz.keys())}
        else:
            self._landm_regressors = {k: (v, np.array([1.0])) for k, v in self.landm.items()}

    @property
    def landm_names(self):
        if self.landm_regressors is not None:
            return self.landm_regressors.keys()
        elif self.landm is not None:
            return self.landm.keys()
        else:
            return []

    def compute_landmark_indices(self, landm_xyz):
        import numpy as np
        if len(landm_xyz) > 0:
            closest_vertices, _ = self.closest_vertices(np.array(landm_xyz.values()))
            return dict(zip(landm_xyz.keys(), closest_vertices))
        else:
            return {}

    def set_landmarks_from_xyz(self, landm_raw_xyz):
        raise DeprecationWarning("Mesh.set_landmarks_from_xyz is deprecated in favor of just setting mesh.landm_xyz directly")

    def set_landmark_indices_from_any(self, landmarks):
        raise DeprecationWarning("Mesh.set_landmark_indices_from_any is deprecated in favor of just setting mesh.landm directly")

    def set_landmarks_from_raw(self, landmarks):
        raise DeprecationWarning("Mesh.set_landmarks_from_raw is deprecated in favor of just setting mesh.landm directly")

    def set_landmarks_from_regressors(self, regressors):
        raise DeprecationWarning("Mesh.set_landmarks_from_regressors is deprecated in favor of just setting mesh.landm_regressors directly")
