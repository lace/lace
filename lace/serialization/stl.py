__all__ = ['load', 'dump', 'EXTENSION']

EXTENSION = '.stl'


def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', mesh=existing_mesh)


def dump(mesh, f, name="mesh", ascii=True):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _dump, mode='wb', mesh=mesh, name=name, ascii=ascii)


def _load(f, mesh=None):
    import numpy as np
    from lace.mesh import Mesh
    if not mesh:
        mesh = Mesh()
    faces = []
    facenormals = []
    verts = []

    head = f.readline().strip()
    if head.startswith("solid"): # ascii STL format
        #name = head[6:]
        current_face = []
        for line in f:
            line = line.split()
            if line[0] == "endsolid":
                break
            elif line[0] == "facet":
                current_face = []
                if line[1] == "normal":
                    try:
                        facenormals.append([float(x) for x in line[2:]])
                    except: # pylint: disable=bare-except
                        facenormals.append([np.nan, np.nan, np.nan])
                else:
                    facenormals.append([np.nan, np.nan, np.nan])
            elif line[0] == "endfacet":
                faces.append(current_face)
                current_face = []
            elif line[0:2] == ["outer", "loop"]:
                pass
            elif line[0] == "endloop":
                pass
            elif line[0] == "vertex":
                current_face.append(len(verts))
                try:
                    verts.append([float(x) for x in line[1:]])
                except: # pylint: disable=bare-except
                    verts.append([np.nan, np.nan, np.nan])
            else:
                raise ValueError("Badly formatted STL file. I don't understand the line %s" % line)
    else:
        raise Exception("Looks like this is a binary STL file; you're going to have to implement that")
        # format docs are here: http://en.wikipedia.org/wiki/STL_(file_format)

    mesh.v = np.array(verts, dtype=np.float64).copy()
    mesh.f = np.array(faces, dtype=np.uint32).copy()
    mesh.fn = np.array(facenormals, dtype=np.float64).copy()
    return mesh


def _dump(f, mesh, name="mesh", ascii=True): # pylint: disable=unused-argument
    raise NotImplementedError("stl.dump is not implemented yet")
