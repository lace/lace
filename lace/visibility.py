class MeshMixin(object):
    def vertex_visibility(self, camera, normal_threshold=None, omni_directional_camera=False, binary_visiblity=True):
        import numpy as np
        vis, n_dot_cam = self.vertex_visibility_and_normals(camera, omni_directional_camera)
        if normal_threshold is not None:
            vis = np.logical_and(vis, n_dot_cam > normal_threshold)
        return np.squeeze(vis) if binary_visiblity else np.squeeze(vis*n_dot_cam)

    def vertex_visibility_and_normals(self, camera, omni_directional_camera=False):
        import numpy as np
        from lace_search.visibility import visibility_compute # pylint: disable=no-name-in-module
        arguments = {'v': self.v, 'f': self.f, 'cams' : np.array([camera.origin.flatten()])}
        if not omni_directional_camera:
            arguments['sensors'] = np.array([camera.sensor_axis.flatten()])
        arguments['n'] = self.vn if hasattr(self, 'vn') else self.estimate_vertex_normals()

        return visibility_compute(**arguments)

    def visibile_mesh(self, camera=[0.0, 0.0, 0.0]):
        import numpy as np
        vis = self.vertex_visibility(camera)
        faces_to_keep = filter(lambda face: vis[face[0]]*vis[face[1]]*vis[face[2]], self.f)
        vertex_indices_to_keep = np.nonzero(vis)[0]
        vertices_to_keep = self.v[vertex_indices_to_keep]
        old_to_new_indices = np.zeros(len(vis))
        old_to_new_indices[vertex_indices_to_keep] = range(len(vertex_indices_to_keep))
        return self.__class__(v=vertices_to_keep, f=np.array([old_to_new_indices[face] for face in faces_to_keep]))
