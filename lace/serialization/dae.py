__all__ = ['dump']

def dump(mesh, f, e_color=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _dump, 'w', mesh, e_color=e_color)

def _dump(f, mesh, e_color=None):
    '''
    Writes a mesh to collada file format.
    '''
    dae = mesh_to_collada(mesh, e_color=e_color)
    dae.write(f.name)

def dumps(mesh, e_color=None):
    '''
    Generates a UTF-8 XML string containing the mesh, in collada format.
    '''
    from lxml import etree

    dae = mesh_to_collada(mesh, e_color=e_color)

    # Update the xmlnode.
    dae.save()

    return etree.tostring(dae.xmlnode, encoding='UTF-8')

def mesh_to_collada(mesh, e_color=None):
    '''
    Supports per-vertex color, but nothing else.
    '''
    import numpy as np
    try:
        from collada import Collada, scene
    except ImportError:
        raise ImportError("lace.serialization.dae.mesh_to_collade requires package pycollada.")


    def create_material(dae, name, color=(1, 1, 1)):
        from collada import material, scene
        effect = material.Effect("{}_effect".format(name), [], "lambert", diffuse=color, specular=(0, 0, 0), double_sided=True)
        mat = material.Material("{}_material".format(name), name, effect)
        dae.effects.append(effect)
        dae.materials.append(mat)
        return scene.MaterialNode(name, mat, inputs=[])

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
        triset = geom.createTriangleSet(indices, input_list, "tri_material")
        geom.primitives.append(triset)

        extra_materials = []
        # e
        if mesh.e is not None:
            if e_color is None:
                indices = np.dstack([mesh.e for _ in srcs]).ravel()
                lineset = geom.createLineSet(indices, input_list, "line_material")
                geom.primitives.append(lineset)
            else:
                edges_rendered = np.zeros(len(mesh.e), dtype=np.bool)
                for i, this_e_color in enumerate(e_color):
                    these_edge_indices = this_e_color["e_indices"]
                    this_color = this_e_color["color"]
                    material_name = "line_material_{}".format(i)
                    indices = np.dstack(
                        [mesh.e[these_edge_indices] for _ in srcs]
                    ).ravel()
                    extra_materials.append(
                        create_material(dae, name=material_name, color=this_color)
                    )
                    lineset = geom.createLineSet(indices, input_list, material_name)
                    geom.primitives.append(lineset)
                    edges_rendered[these_edge_indices] = True
                edges_remaining = (~edges_rendered).nonzero()
                if len(edges_remaining):
                    indices = np.dstack([mesh.e[edges_remaining] for _ in srcs]).ravel()
                    lineset = geom.createLineSet(indices, input_list, "line_material")
                    geom.primitives.append(lineset)

        dae.geometries.append(geom)
        return geom, extra_materials

    dae = Collada()
    geom, extra_materials = geometry_from_mesh(dae, mesh)
    node = scene.Node(
        "node0",
        children=[
            scene.GeometryNode(
                geom,
                [
                    create_material(dae, name="tri_material"),
                    create_material(dae, name="line_material", color=(1, 0, 0)),
                ]
                + extra_materials,
            )
        ],
    )
    myscene = scene.Scene("myscene", [node])
    dae.scenes.append(myscene)
    dae.scene = myscene
    return dae
