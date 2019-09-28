# pylint: disable=attribute-defined-outside-init
class MeshMixin(object):

    @property
    def texture_filepath(self):
        if getattr(self, '_texture_filepath', None) is None:
            return None
        if self._texture_filepath.startswith('/is/ps/'):
            # This is a special case to support some old meshes, scape templates in particular,
            # that have old MPI paths encoded in them. Note that we do this here rather than in
            # the setter, since some of them we're loading from pickle files and unpickling doesn't
            # call the setter.
            return "s3://bodylabs-korper-assets" + self._texture_filepath
        return self._texture_filepath

    @texture_filepath.setter
    def texture_filepath(self, val):
        self._texture_image = None # To ensure reloading
        self._texture_filepath = val

    @property
    def texture_image(self):
        if getattr(self, '_texture_image', None) is None:
            self.reload_texture_image()
        return self._texture_image

    def set_texture_image(self, path_to_texture):
        self.texture_filepath = path_to_texture

    def texture_coordinates_by_vertex(self):
        texture_coordinates_by_vertex = [[] for i in range(len(self.v))]
        for i, face in enumerate(self.f):
            for j in [0, 1, 2]:
                texture_coordinates_by_vertex[face[j]].append(self.vt[self.ft[i][j]])
        return texture_coordinates_by_vertex


    def reload_texture_image(self):
        import cv2
        import numpy as np
        from baiji import s3
        if not self.texture_filepath:
            self._texture_image = None
        else:
            # image is loaded as image_height-by-image_width-by-3 array in BGR color order.
            with s3.open(self.texture_filepath) as f:
                self._texture_image = cv2.imread(f.name)
            texture_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
            if self._texture_image is not None:
                h, w = self._texture_image.shape[:2]
                if h != w or h not in texture_sizes or w not in texture_sizes:
                    closest_texture_size_idx = (np.abs(np.array(texture_sizes) - max(self._texture_image.shape))).argmin()
                    sz = texture_sizes[closest_texture_size_idx]
                    self._texture_image = cv2.resize(self._texture_image, (sz, sz))

    def transfer_texture(self, mesh_with_texture):
        import numpy as np
        if not np.all(mesh_with_texture.f.shape == self.f.shape):
            raise Exception('Mesh topology mismatch')
        self.vt = mesh_with_texture.vt.copy()
        self.ft = mesh_with_texture.ft.copy()
        if not np.all(mesh_with_texture.f == self.f):
            if np.all(mesh_with_texture.f == np.fliplr(self.f)):
                self.ft = np.fliplr(self.ft)
            else:
                # Same shape; let's see if it's face ordering; this could be a bit faster...
                face_mapping = {}
                for f, ii in zip(self.f, range(len(self.f))):
                    face_mapping[" ".join([str(x) for x in sorted(f)])] = ii
                self.ft = np.zeros(self.f.shape, dtype=np.uint32)
                for f, ft in zip(mesh_with_texture.f, mesh_with_texture.ft):
                    k = " ".join([str(x) for x in sorted(f)])
                    if k not in face_mapping:
                        raise Exception('Mesh topology mismatch')
                    # the vertex order can be arbitrary...
                    ids = []
                    for f_id in f:
                        ids.append(np.where(self.f[face_mapping[k]] == f_id)[0][0])
                    ids = np.array(ids)
                    self.ft[face_mapping[k]] = np.array(ft[ids])
        self.texture_filepath = mesh_with_texture.texture_filepath
        self._texture_image = None

    def texture_rgb(self, texture_coordinate):
        import numpy as np
        h, w = np.array(self.texture_image.shape[:2]) - 1
        return np.double(self.texture_image[int(h*(1.0 - texture_coordinate[1]))][int(w*(texture_coordinate[0]))])[::-1]

    def texture_rgb_vec(self, texture_coordinates):
        import numpy as np
        h, w = np.array(self.texture_image.shape[:2]) - 1
        n_ch = self.texture_image.shape[2]
        # XXX texture_coordinates can be lower than 0! clip needed!
        d1 = (h*(1.0 - np.clip(texture_coordinates[:, 1], 0, 1))).astype(np.int)
        d0 = (w*(np.clip(texture_coordinates[:, 0], 0, 1))).astype(np.int)
        flat_texture = self.texture_image.flatten()
        indices = np.hstack([((d1*(w+1)*n_ch) + (d0*n_ch) + (2-i)).reshape(-1, 1) for i in range(n_ch)])
        return flat_texture[indices]
