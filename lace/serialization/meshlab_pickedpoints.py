'''
Reads and writes a Meshlab picked points file. Files contain named
points in 3D space. A set of picked points can be saved alongside
the mesh and loaded in Meshlab (on the Edit menu, click PickPoints),
or via Mesh(filename=mesh_filename, landmarks=pp_filename).

'''

from __future__ import absolute_import
__all__ = ['load', 'dump', 'loads', 'dumps', 'EXTENSION']

EXTENSION = '.pp'


def dump(obj, f, *args, **kwargs):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _dump, 'w', obj, *args, **kwargs)


def load(f, *args, **kwargs):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, 'r', *args, **kwargs)


def loads(s, *args, **kwargs): # pylint: disable=unused-argument
    # TODO This seems tricky with xml.etree, probably easiest to
    # write the file and then read it again?
    raise NotImplementedError()


def dumps(obj, mesh_filename=None, *args, **kwargs): # pylint: disable=unused-argument
    '''
    obj: A dictionary mapping names to a 3-dimension array.
    mesh_filename: If provided, this value is included in the <DataFileName>
      attribute, which Meshlab doesn't seem to use.

    TODO Maybe reconstruct this using xml.etree

    '''
    point_template = '<point x="%f" y="%f" z="%f" name="%s"/>\n'
    file_template = """
    <!DOCTYPE PickedPoints>
    <PickedPoints>
     <DocumentData>
      <DateTime time="16:00:00" date="2014-12-31"/>
      <User name="bodylabs"/>
      <DataFileName name="%s"/>
     </DocumentData>
     %s
    </PickedPoints>
    """
    from blmath.numerics import isnumericarray

    if not isinstance(obj, dict) or not all([isnumericarray(point) for point in obj.itervalues()]):
        raise ValueError('obj should be a dict of points')

    points = '\n'.join([point_template % (tuple(xyz) + (name,)) for name, xyz in obj.iteritems()])

    return file_template % (mesh_filename, points)


def _dump(f, obj, *args, **kwargs):
    xml_string = dumps(obj, *args, **kwargs)
    f.write(xml_string)


def _load(f, *args, **kwargs): # pylint: disable=unused-argument
    from xml.etree import ElementTree
    tree = ElementTree.parse(f)

    points = {}

    for e in tree.iter('point'):
        try:
            point = [float(e.attrib['x']), float(e.attrib['y']), float(e.attrib['z'])]
        except ValueError: # may happen if landmarks are just spaces
            point = [0, 0, 0]
        points[e.attrib['name']] = point

    return points
