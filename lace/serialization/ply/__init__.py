from lace.serialization.ply.plyutils import error as PLYError # pylint can't find the exception type in the c code pylint: disable=no-name-in-module

__all__ = ['load', 'dump', 'EXTENSION']

EXTENSION = '.ply'


def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', mesh=existing_mesh)


def dump(mesh, f, flip_faces=False, ascii=False, little_endian=True, comments=None): # pylint: disable=redefined-builtin
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    if comments is None:
        comments = []
    return ensure_file_open_and_call(f, _dump, mode='wb', mesh=mesh, flip_faces=flip_faces, ascii=ascii, little_endian=little_endian, comments=comments)

def _load(f, mesh=None):
    import numpy as np
    from lace.mesh import Mesh
    from lace.serialization.ply import plyutils
    res = plyutils.read(f.name)
    if not mesh:
        mesh = Mesh()
    mesh.v = np.array(res['pts'], dtype=np.float64).T.copy()
    mesh.f = np.array(res['tri'], dtype=np.uint32).T.copy()

    if not mesh.f.shape[0]:
        mesh.f = None

    if 'color' in res:
        mesh.vc = np.array(res['color']).T.copy() / 255
    if 'normals' in res:
        mesh.vn = np.array(res['normals']).T.copy()
    return mesh


def _dump(f, mesh, flip_faces=False, ascii=False, little_endian=True, comments=[]):
    # pylint: disable=superfluous-parens
    from lace.serialization.ply import plyutils
    ff = -1 if flip_faces else 1
    if isinstance(comments, basestring):
        comments = [comments]
    comments = filter(lambda c: len(c) > 0, sum(map(lambda c: c.split("\n"), comments), []))
    plyutils.write(list([list(x) for x in mesh.v]),
                   list([list(x[::ff]) for x in mesh.f] if mesh.f is not None else []),
                   list([list((x*255).astype(int)) for x in (mesh.vc if mesh.vc is not None else [])]),
                   f.name, ascii, little_endian, list(comments),
                   list([list(x) for x in (mesh.vn if mesh.vn is not None else [])]))
