def slimmed_point_cloud(mesh, n_verts_desired=25000):
    '''
    Return a point cloud slimmed down to the desired number of vertices.

    FIXME This blows away the segmentation.

    '''
    import math
    import numpy as np
    from bodylabs.mesh import Mesh

    if mesh.v is None or not mesh.v.shape[0]:
        raise ValueError('Mesh has no vertices')

    if mesh.v.shape[0] > n_verts_desired:
        indexes_to_keep = [int(math.ceil(i*float(len(mesh.v)) / n_verts_desired)) for i in range(n_verts_desired)]
    else:
        indexes_to_keep = range(n_verts_desired)

    verts_to_keep = mesh.v[indexes_to_keep]
    return Mesh(f=np.ndarray((0, 3)), v=verts_to_keep)

def slimmed_mesh(mesh, n_faces_desired=50000):
    '''
    Return a mesh slimmed down to the desired number of faces.

    '''
    if mesh.v is None or not mesh.v.shape[0]:
        raise ValueError('Mesh has no vertices')

    if mesh.f is None or not mesh.f.shape[0]:
        # We estimate n_verts_desired to be half the number of faces desired
        return slimmed_point_cloud(mesh, n_verts_desired=n_faces_desired / 2)
    elif mesh.f.shape[0] > n_faces_desired:
        return qslim(mesh, n_tris=n_faces_desired)
    else:
        return mesh

def qslim(mesh, n_tris=10000, want_optimal=False):
    '''
    Since our qslim command only operates on obj files, we write the mesh out and then run the qslim function
    '''
    from lace.serialization import obj
    from bodylabs.serialization.temporary import Tempfile

    with Tempfile(mesh, obj.dump, suffix='.obj') as tf:
        return qslim_obj_file(tf.name, n_tris, want_optimal)

def qslim_obj_file(path, n_tris=10000, want_optimal=False):
    '''
    A wrapper for the sqlim command, which we don't have source to or a windows version of
    '''
    import os
    import platform
    import sys
    from subprocess import check_output
    from bodylabs.mesh import Mesh

    placement = '3' if want_optimal else '0'
    options = ['-O', placement, '-t', str(n_tris)]
    if platform.system() == 'Darwin':
        qslim_command = 'qslim_osx'
        options.extend(['-m', '1000'])
    elif platform.system() == 'Linux':
        qslim_command = 'qslim_x86_64'
    else:
        raise NotImplementedError("qslim not implemented on windows")
    command = [os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'bin', qslim_command)] + options + [path]
    m_str = check_output(command, stderr=sys.stderr)
    # parse std_out to mesh vertices and faces
    m_lines = m_str.split('\n')
    m_v = [map(float, s[2:].split()) for s in m_lines if s and s[0] == 'v']
    m_f = [map(lambda st: int(st)-1, s[2:].split()) for s in m_lines if s and s[0] == 'f']
    return Mesh(v=m_v, f=m_f)

def write_slimmed_mesh(mesh, filename, with_copyright=False, n_faces_desired=50000):
    '''
    Convenience function for slimmed_mesh. with_copyright is intended for our
    marketplace bodies, where the customer does not own the scan.

    '''
    from lace.serialization import obj
    slimmed = slimmed_mesh(mesh, n_faces_desired=n_faces_desired)
    obj.dump(slimmed, filename, copyright=with_copyright)
