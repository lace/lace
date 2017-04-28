__all__ = ['dump']

def dump(mesh, f):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _dump, 'w', mesh)

def _dump(f, mesh):
    '''
    Writes a mesh to collada file format.
    '''
    dae = mesh_to_collada(mesh)
    dae.write(f.name)

def dumps(mesh):
    '''
    Generates a UTF-8 XML string containing the mesh, in collada format.
    '''
    from lxml import etree

    dae = mesh_to_collada(mesh)

    # Update the xmlnode.
    dae.save()

    return etree.tostring(dae.xmlnode, encoding='UTF-8')

def mesh_to_collada(mesh):
    '''
    Supports per-vertex color, but nothing else.
    '''
    import numpy as np
    try:
        from collada import Collada, scene
    except ImportError:
        raise ImportError("lace.serialization.dae.mesh_to_collade requires package pycollada.")


    def create_material(dae):
        from collada import material, scene
        effect = material.Effect("effect0", [], "phong", diffuse=(1, 1, 1), specular=(0, 0, 0), double_sided=True)
        mat = material.Material("material0", "mymaterial", effect)
        dae.effects.append(effect)
        dae.materials.append(mat)
        return scene.MaterialNode("materialref", mat, inputs=[])

    def geometry_from_mesh(dae, mesh):
        from collada import source, geometry
        srcs = []
        # v
        srcs.append(source.FloatSource("verts-array", mesh.v, ('X', 'Y', 'Z')))
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', "#verts-array")
        # vc
        if mesh.vc is not None:
            input_list.addInput(len(srcs), 'COLOR', "#color-array")
            srcs.append(source.FloatSource("color-array", mesh.vc[mesh.f.ravel()], ('X', 'Y', 'Z')))
        # f
        geom = geometry.Geometry(str(mesh), "geometry0", "mymesh", srcs)
        indices = np.dstack([mesh.f for _ in srcs]).ravel()
        triset = geom.createTriangleSet(indices, input_list, "materialref")
        geom.primitives.append(triset)
        # e
        if mesh.e is not None:
            indices = np.dstack([mesh.e for _ in srcs]).ravel()
            lineset = geom.createLineSet(indices, input_list, "materialref")
            geom.primitives.append(lineset)
        dae.geometries.append(geom)
        return geom

    dae = Collada()
    geom = geometry_from_mesh(dae, mesh)
    node = scene.Node("node0", children=[scene.GeometryNode(geom, [create_material(dae)])])
    myscene = scene.Scene("myscene", [node])
    dae.scenes.append(myscene)
    dae.scene = myscene
    return dae
