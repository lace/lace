'''
Reads the CAESAR .lmrk file format.

'''

from __future__ import absolute_import
__all__ = ['load', 'loads']

def load(f, *args, **kwargs):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, 'rb', *args, **kwargs)

def _load(f, *args, **kwargs): # pylint: disable=unused-argument
    return loads(f.read())

def loads(s, ret_commands=False):
    import numpy as np

    landmarks = {}
    commands = {}

    for line in s.splitlines():
        if not line.strip():
            continue

        key = line.split()[0]
        data = [float(x) for x in line.split()[1:]]

        if key == '_scale':
            commands['scale'] = np.matrix(data)
        elif key == '_translate':
            commands['translate'] = np.matrix(data)
        elif key == '_rotation':
            commands['rotation'] = np.matrix(data).reshape(3, 3)
        else:
            landmarks[key] = [data[1], data[2], data[0]]

    return (landmarks, commands) if ret_commands else landmarks
