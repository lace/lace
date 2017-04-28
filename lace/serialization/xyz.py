def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', existing_mesh=existing_mesh)

def _load(f, existing_mesh=None):
    import numpy as np
    from lace.mesh import Mesh
    v = np.loadtxt(f.name)
    if existing_mesh is None:
        return Mesh(v=v)
    else:
        existing_mesh.v = v
        return existing_mesh
