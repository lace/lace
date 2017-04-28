# pylint: disable=len-as-condition
__all__ = ['load', 'ParseError']

class ParseError(Exception):
    pass

def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', mesh=existing_mesh)

def _load(f, mesh=None):
    from lace.mesh import Mesh

    try:
        v, f, vn, vc = _parse_wrl(f)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        raise ParseError("Unable to parse wrl file with exception : \n[\n%s]" % tb)

    if not mesh:
        mesh = Mesh()
    if v.size != 0:
        mesh.v = v
    if f.size != 0:
        mesh.f = f
    if vn.size != 0:
        mesh.vn = vn
    if vc.size != 0:
        mesh.vc = vc
    return mesh

def _find_next_block(f, keywords=['coord', 'color', 'normal', 'coordIndex']):
    for line in f:
        line_splitted = line.split()
        if not line_splitted:
            continue
        if line_splitted[0] in keywords:
            return line_splitted[0]

    return None

def _load_faces(f, target, start_index):
    for face_line in f:
        if not face_line.strip():
            continue
        i = face_line.strip().find('-1')
        if i < 0:
            break
        target.append(map(lambda x: int(x) + start_index, face_line.split(',')[:3]))

def _load_per_vertex_data(f, target):
    for vertex_line in f:

        if not vertex_line.strip():
            continue

        i = vertex_line.strip().find(',')
        if i < 0:
            break
        target.append(map(float, vertex_line.strip()[:i].split()))

def _parse_wrl(f):

    import numpy as np
    v = []
    vn = []
    vc = []
    faces = []
    start_index = 0

    format_disclaimer = f.readline()
    if format_disclaimer.strip() != '#VRML V2.0 utf8':
        raise ParseError("Unsupported wrl file format")

    while True:
        key = _find_next_block(f)
        if key == 'coord':
            _load_per_vertex_data(f, v)
        elif key == 'color':
            _load_per_vertex_data(f, vc)
        elif key == 'normal':
            _load_per_vertex_data(f, vn)
        elif key == 'coordIndex':
            _load_faces(f, faces, start_index)
            start_index = len(v)
        else:
            break

    if len(v) == 0:
        raise ParseError("Unsupported wrl file format")
    return np.array(v), np.array(faces), np.array(vn), np.array(vc)
