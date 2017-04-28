class MeshMixin(object):
    '''
    Convenience methods for serialization.
    Please write serializers rather than adding explicit serialization code in here.
    '''

    def write(self, filename):
        import os
        _, fileext = os.path.splitext(filename)
        if fileext == '.obj':
            self.write_obj(filename)
        elif fileext == '.ply':
            self.write_ply(filename)
        elif fileext == '.dae':
            self.write_dae(filename)
        else:
            raise TypeError('Unsupported filetype %s' % fileext)

    def write_obj(self, filename, flip_faces=False, ungroup=False, comments=None, write_mtl=True):
        from lace.serialization import obj
        obj.dump(self, filename, flip_faces=flip_faces, ungroup=ungroup, comments=comments, write_mtl=write_mtl)

    def write_fuse_obj(self, filename):
        from lace.serialization import obj
        obj.dump(self, filename, split_normals=True)

    def write_mtl(self, path, material_name, texture_name):
        from lace.serialization import obj
        obj.write_mtl(path, material_name, texture_name)

    def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
        from lace.serialization import ply
        ply.dump(self, filename, flip_faces=flip_faces, ascii=ascii, little_endian=little_endian, comments=comments)

    def write_dae(self, filename):
        from lace.serialization import dae
        dae.dump(self, filename)
