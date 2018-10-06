from __future__ import print_function
import numpy as np


def faces_with_repeated_vertices(f):
    if f.shape[1] == 3:
        return np.unique(np.concatenate([
            np.where(f[:, 0] == f[:, 1])[0],
            np.where(f[:, 0] == f[:, 2])[0],
            np.where(f[:, 1] == f[:, 2])[0],
        ]))
    else:
        return np.unique(np.concatenate([
            np.where(f[:, 0] == f[:, 1])[0],
            np.where(f[:, 0] == f[:, 2])[0],
            np.where(f[:, 0] == f[:, 3])[0],
            np.where(f[:, 1] == f[:, 2])[0],
            np.where(f[:, 1] == f[:, 3])[0],
            np.where(f[:, 2] == f[:, 3])[0],
        ]))

def faces_with_out_of_range_vertices(f, v):
    return np.unique(np.concatenate([
        np.where(f < 0)[0],
        np.where(f >= len(v))[0],
    ]))

def check_integrity(mesh):
    errors = []
    for f_index in faces_with_out_of_range_vertices(mesh.f, mesh.v):
        errors.append(("f", f_index, "Vertex out of range"))
    for f_index in faces_with_repeated_vertices(mesh.f):
        errors.append(("f", f_index, "Repeated vertex"))
    return errors

def print_integrity_errors(errors, mesh):
    for attr, index, message in errors:
        try:
            data = getattr(mesh, attr)[index]
        except (AttributeError, IndexError):
            data = ''
        print("{} {}   {}   {}".format(attr, index, message, data))
