from lace.serialization.obj.objutils import LoadObjError # lint isn't able to find the defintion in a c++ module pylint: disable=no-name-in-module

EXTENSION = '.obj'

def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='r', mesh=existing_mesh)

def dump(obj, f, flip_faces=False, ungroup=False, comments=None,
         copyright=False, split_normals=False, write_mtl=True): # pylint: disable=redefined-outer-name, redefined-builtin, unused-argument
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    if comments is None:
        comments = []
    return ensure_file_open_and_call(f, _dump, mode='w', obj=obj, flip_faces=flip_faces,
                                     ungroup=ungroup, comments=comments,
                                     split_normals=split_normals, write_mtl=write_mtl)

def _load(fd, mesh=None):
    from collections import OrderedDict
    from baiji import s3
    from lace.mesh import Mesh
    import lace.serialization.obj.objutils as objutils # pylint: disable=no-name-in-module

    v, vt, vn, vc, f, ft, fn, mtl_path, landm, segm = objutils.read(fd.name)
    if not mesh:
        mesh = Mesh()
    if v.size != 0:
        mesh.v = v
    if f.size != 0:
        mesh.f = f
    if vn.size != 0:
        mesh.vn = vn
    if vt.size != 0:
        mesh.vt = vt
    if vc.size != 0:
        mesh.vc = vc
    if fn.size != 0:
        mesh.fn = fn
    if ft.size != 0:
        mesh.ft = ft
    if segm:
        mesh.segm = OrderedDict([(k, v if isinstance(v, list) else v.tolist()) for k, v in segm.items()])
    def path_relative_to_mesh(filename):

        # The OBJ file we're loading may have come from a local path, or an s3
        # url. Since OBJ defines materials and texture files with paths
        # relative to the OBJ itself, we need tocope with the various
        # possibilities and if it's a cached file make sure that the material
        # and texture have been downloaded as well.
        #
        # If an absolute path is given and the file is missing, try looking in
        # the same directory; this lets you find the most common intention when
        # an abs path is used.
        #
        # NB: We do not support loading material & texture info from objs read
        # from filelike objects without a location on the filesystem; what would
        # the relative file names mean in that case, anyway? (unless we're given
        # a valid absolute path, in which case go for it)
        import os
        import re

        # The second term here let's us detect windows absolute paths when we're running on posix
        if filename == os.path.abspath(filename) or re.match(r'^.\:(\\|/)', filename):
            if s3.exists(filename):
                return filename
            else:
                filename = s3.path.basename(filename)

        if hasattr(fd, 'remotename'):
            mesh_path = fd.remotename
        elif hasattr(fd, 'name'):
            mesh_path = fd.name
        else:
            return None

        path = s3.path.join(s3.path.dirname(mesh_path), filename)
        return path

    mesh.materials_filepath = None
    if mtl_path:
        materials_filepath = path_relative_to_mesh(mtl_path.strip())
        if materials_filepath and s3.exists(materials_filepath):
            with s3.open(materials_filepath, 'r') as f:
                mesh.materials_file = f.readlines()
            mesh.materials_filepath = materials_filepath

    if hasattr(mesh, 'materials_file'):
        mesh.texture_filepaths = {
            line.split(None, 1)[0].strip(): path_relative_to_mesh(line.split(None, 1)[1].strip())
            for line in mesh.materials_file if line.startswith('map_K')
        }
        if 'map_Ka' in mesh.texture_filepaths:
            mesh.texture_filepath = mesh.texture_filepaths['map_Ka']
        elif 'map_Kd' in mesh.texture_filepaths:
            mesh.texture_filepath = mesh.texture_filepaths['map_Kd']

    if landm:
        mesh.landm = landm
    return mesh

def _dump(f, obj, flip_faces=False, ungroup=False, comments=None, split_normals=False, write_mtl=True): # pylint: disable=redefined-outer-name
    '''
    write_mtl: When True and mesh has a texture, includes a mtllib
      reference in the .obj and writes a .mtl alongside.

    '''
    import six
    import os
    import numpy as np
    from baiji import s3

    ff = -1 if flip_faces else 1
    def write_face_to_obj_file(obj, faces, face_index, obj_file):
        vertex_indices = faces[face_index][::ff] + 1

        write_normals = obj.fn is not None or (obj.vn is not None and obj.vn.shape == obj.v.shape)
        write_texture = obj.ft is not None and obj.vt is not None

        if write_normals and obj.fn is not None:
            normal_indices = obj.fn[face_index][::ff] + 1
            assert len(normal_indices) == len(vertex_indices)
        elif write_normals: # unspecified fn but per-vertex normals, assume ordering is same as for v
            normal_indices = faces[face_index][::ff] + 1

        if write_texture:
            texture_indices = obj.ft[face_index][::ff] + 1
            assert len(texture_indices) == len(vertex_indices)

        # Valid obj face lines are: v, v/vt, v//vn, v/vt/vn
        if write_normals and write_texture:
            pattern = '%d/%d/%d'
            value = tuple(np.array([vertex_indices, texture_indices, normal_indices]).T.flatten())
        elif write_normals:
            pattern = '%d//%d'
            value = tuple(np.array([vertex_indices, normal_indices]).T.flatten())
        elif write_texture:
            pattern = '%d/%d'
            value = tuple(np.array([vertex_indices, texture_indices]).T.flatten())
        else:
            pattern = '%d'
            value = tuple(vertex_indices)
        obj_file.write(('f ' + ' '.join([pattern]*len(vertex_indices)) + '\n') % value)

    if comments != None:
        if isinstance(comments, six.string_types):
            comments = [comments]
        for comment in comments:
            for line in comment.split("\n"):
                f.write("# %s\n" % line)

    if write_mtl and hasattr(obj, 'texture_filepath') and obj.texture_filepath is not None:
        save_to = s3.path.dirname(f.name)
        mtl_name = os.path.splitext(s3.path.basename(f.name))[0]
        mtl_filename = mtl_name + '.mtl'
        f.write('mtllib %s\n' % mtl_filename)
        f.write('usemtl %s\n' % mtl_name)
        texture_filename = mtl_name + os.path.splitext(obj.texture_filepath)[1]
        if not s3.exists(s3.path.join(save_to, texture_filename)):
            s3.cp(obj.texture_filepath, s3.path.join(save_to, texture_filename))
        obj.write_mtl(s3.path.join(save_to, mtl_filename), mtl_name, texture_filename)

    if obj.vc is not None:
        for r, c in zip(obj.v, obj.vc):
            f.write('v %f %f %f %f %f %f\n' % (r[0], r[1], r[2], c[0], c[1], c[2]))
    elif obj.v is not None:
        for r in obj.v:
            f.write('v %f %f %f\n' % (r[0], r[1], r[2]))

    if obj.vn is not None:
        if split_normals:
            for vn_idx in obj.fn:
                r = obj.vn[vn_idx[0]]
                f.write('vn %f %f %f\n' % (r[0], r[1], r[2]))
                r = obj.vn[vn_idx[1]]
                f.write('vn %f %f %f\n' % (r[0], r[1], r[2]))
                r = obj.vn[vn_idx[2]]
                f.write('vn %f %f %f\n' % (r[0], r[1], r[2]))
        else:
            for r in obj.vn:
                f.write('vn %f %f %f\n' % (r[0], r[1], r[2]))

    if obj.ft is not None and obj.vt is not None:
        for r in obj.vt:
            if len(r) == 3:
                f.write('vt %f %f %f\n' % (r[0], r[1], r[2]))
            else:
                f.write('vt %f %f\n' % (r[0], r[1]))
    if obj.f4 is not None:
        faces = obj.f4
    elif obj.f is not None:
        faces = obj.f
    else:
        faces = None
    if obj.segm is not None and not ungroup:
        if faces is not None:
            # An array of strings.
            group_names = np.array(obj.segm.keys())

            # A 2d array of booleans indicating which face is in which group.
            group_mask = np.zeros((len(group_names), len(faces)), dtype=bool)
            for i, segm_faces in enumerate(obj.segm.itervalues()):
                group_mask[i][segm_faces] = True

            # In an OBJ file, "g" changes the current state. This is a slice of
            # group_mask that represents the current state.
            current_group_mask = np.zeros((len(group_names),), dtype=bool)

            for face_index in range(len(faces)):
                # If the group has changed from the previous face, write the
                # group entry.
                this_group_mask = group_mask[:, face_index]
                if any(current_group_mask != this_group_mask):
                    current_group_mask = this_group_mask
                    f.write('g %s\n' % ' '.join(group_names[current_group_mask]))

                write_face_to_obj_file(obj, faces, face_index, f)
    else:
        if faces is not None:
            for face_index in range(len(faces)):
                write_face_to_obj_file(obj, faces, face_index, f)

def write_mtl(path, material_name, texture_name):
    from baiji import s3
    with s3.open(path, 'w') as f:
        f.write('newmtl %s\n' % material_name)
        # copied from another obj, no idea about what it does
        f.write('ka 0.329412 0.223529 0.027451\n')
        f.write('kd 0.780392 0.568627 0.113725\n')
        f.write('ks 0.992157 0.941176 0.807843\n')
        f.write('illum 0\n')
        f.write('map_Ka %s\n'%texture_name)
        f.write('map_Kd %s\n'%texture_name)
        f.write('map_Ks %s\n'%texture_name)
