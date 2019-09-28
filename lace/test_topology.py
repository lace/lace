# pylint: disable=len-as-condition
import unittest
import mock
import numpy as np
import scipy.sparse as sp
from bltest import attr
from lace.mesh import Mesh
from lace.serialization import obj
from lace.cache import vc

class TestTopologyMixin(unittest.TestCase):

    def test_quads_to_tris(self):
        from lace.topology import quads_to_tris

        tris = np.array([
            [3, 2, 1, 0],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
        ])
        expected_quads = np.array([
            [3, 2, 1],
            [3, 1, 0],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
        ])
        expected_f_old_to_new = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
        ])
        np.testing.assert_array_equal(quads_to_tris(tris), expected_quads)

        quads, f_old_to_new = quads_to_tris(tris, ret_mapping=True)
        np.testing.assert_array_equal(expected_quads, quads)
        np.testing.assert_array_equal(f_old_to_new, expected_f_old_to_new)

    def indicies_for_testing_keep_vertices(self, mesh):
        '''
        These tests have failed in the past due to hard coded verticies, so let's
        generate what we need programatically.

        keep_vertices will update both .v and .f. The behavior on .v is simple: it
        will remove all vertices except those who's indices it's given. Here we're
        producing indices_to_keep to be the input to keep_vertices and expected_verts,
        which should be the value of .v after the call.

        The behavior on .f is more complex. .f is rewritten, to include only those
        triangles from the original mesh where all three of the vertices are kept. In
        order to test this we're building indices_to_keep such that it includes all
        the vertices of several known triangles and a bunch of vertices that are chosen
        such that they do not span any triangles. This way we can be sure that the
        function is discarding and keeping faces appropriately. To test this, we look
        at expected_face_vertices, which is the subset of expected_verts that are
        attached to faces. If this were to include all of expected_verts then we would
        know that incomplete faces were being improperly included.
        '''
        expected_faces = [300, 900] # These are arbitrary.
        indices_to_keep = list(mesh.f[expected_faces].flatten())
        faces_seen = set(expected_faces)
        v = 0
        num_inds_to_keep = len(indices_to_keep) + 6 # Doesn't really matter how many, as long as there's a few
        while len(indices_to_keep) < num_inds_to_keep and v < mesh.v.shape[0]:
            # Adding a bunch of vertices that are chosen such that they do not span any triangles:
            faces_containing_v = set(np.nonzero(np.any(mesh.f == v, axis=1))[0])
            if len(faces_containing_v.intersection(faces_seen)) == 0:
                indices_to_keep.append(v)
                faces_seen.update(faces_containing_v)
            v += 1
        expected_verts = mesh.v[np.array(indices_to_keep, dtype=np.uint32)]
        expected_face_vertices = mesh.v[mesh.f[expected_faces].flatten()]
        return indices_to_keep, expected_verts, expected_face_vertices

    @attr('missing_assets')
    def test_keep_vertices(self):
        mesh = obj.load(vc('/templates/cached_model_templates/sm_2013_f_0005.obj'))

        # set vc and vc for completeness
        mesh.set_vertex_colors("blue")
        mesh.reset_normals()

        indices_to_keep, expected_verts, expected_face_vertices = self.indicies_for_testing_keep_vertices(mesh)

        mesh.keep_vertices(indices_to_keep)

        np.testing.assert_array_equal(mesh.v, expected_verts)

        np.testing.assert_array_equal(mesh.v[mesh.f.flatten()], expected_face_vertices)

        max_v_index = np.max(mesh.f.flatten())
        self.assertLessEqual(max_v_index, mesh.v.shape[0] - 1)

    @attr('missing_assets')
    def test_keep_vertices_without_segm(self):
        mesh = obj.load(vc('/templates/cached_model_templates/sm_2013_f_0005.obj'))
        mesh.segm = None

        indices_to_keep, expected_verts, expected_face_vertices = self.indicies_for_testing_keep_vertices(mesh)

        mesh.keep_vertices(indices_to_keep)

        np.testing.assert_array_equal(mesh.v, expected_verts)

        np.testing.assert_array_equal(mesh.v[mesh.f.flatten()], expected_face_vertices)

        max_v_index = np.max(mesh.f.flatten())
        self.assertLessEqual(max_v_index, mesh.v.shape[0] - 1)

    @attr('missing_assets')
    def test_keep_vertices_without_f(self):
        mesh = obj.load(vc('/templates/cached_model_templates/sm_2013_f_0005.obj'))
        mesh.segm = None
        mesh.f = None

        indices_to_keep = [1, 2, 3, 5, 8, 273, 302, 11808, 11847, 12031, 12045]

        expected_verts = mesh.v[indices_to_keep]

        mesh.keep_vertices(indices_to_keep)

        np.testing.assert_array_equal(mesh.v, expected_verts)

        self.assertIs(mesh.f, None)

    def test_keep_vertices_with_no_verts_does_not_raise(self):
        mesh = Mesh()
        mesh.keep_vertices([])

    @attr('missing_assets')
    @mock.patch('warnings.warn')
    def test_keep_vertices_with_empty_list_does_not_warn(self, warn):
        mesh = obj.load(vc('/templates/cached_model_templates/sm_2013_f_0005.obj'))

        mesh.keep_vertices([])

        self.assertFalse(warn.called)

    def test_vertex_indices_in_segments(self):
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        cube.segm = {
            # All quads.
            'all': np.arange(12),
            # Quads 2 and 3.
            'two_adjacent_sides': [4, 5, 6, 7],
            # Quad 0.
            'lower_base': [0, 1],
        }

        np.testing.assert_array_equal(
            cube.vertex_indices_in_segments(['all']),
            np.arange(8)
        )

        np.testing.assert_array_equal(
            len(cube.vertex_indices_in_segments(['lower_base'])),
            4
        )

        np.testing.assert_array_equal(
            len(cube.vertex_indices_in_segments(['two_adjacent_sides'])),
            6
        )

        np.testing.assert_array_equal(
            len(cube.vertex_indices_in_segments(['lower_base', 'two_adjacent_sides'])),
            7
        )

        with self.assertRaises(ValueError):
            cube.vertex_indices_in_segments(['random_segm'])

    @attr('missing_assets')
    def test_keep_segments(self):
        mesh = obj.load(vc('/templates/cached_model_templates/sm_2013_f_0005.obj'))

        expected_parts = ['rightCalf', 'head', 'rightHand', 'leftTorso', 'midsection', 'leftFoot', 'rightTorso', 'rightThigh', 'leftCalf', 'rightShoulder', 'leftShoulder', 'leftThigh', 'pelvis', 'leftForearm', 'rightFoot', 'leftHand', 'rightUpperArm', 'rightForearm', 'leftUpperArm']
        self.assertEqual(set(mesh.segm.keys()), set(expected_parts))

        self.assertEqual(len(mesh.segm['rightFoot']), 3336)
        self.assertEqual(len(mesh.segm['leftFoot']), 3336)

        segments_to_keep = ['leftFoot', 'rightFoot']
        mesh.keep_segments(segments_to_keep)

        self.assertEqual(len(mesh.f), 6672)
        self.assertEqual(len(mesh.segm['rightFoot']), 3336)
        self.assertEqual(len(mesh.segm['leftFoot']), 3336)
        self.assertEqual(set(mesh.segm.keys()), set(segments_to_keep))

        max_f_index = np.max(mesh.segm.values())
        self.assertEqual(max_f_index, mesh.f.shape[0] - 1)

    def test_clean_segments(self):
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        cube.segm = {
            'all': np.arange(12)
        }

        self.assertEqual(cube.clean_segments(['random_segm', 'all']), ['all'])

    def test_flip_faces(self):
        from lace.shapes import create_rectangular_prism
        box = create_rectangular_prism(np.array([1.0, 1.0, 1.0]), np.array([4.0, 2.0, 1.0]))
        box.reset_normals()
        original_vn = box.vn.copy()
        original_f = box.f.copy()
        box.flip_faces()
        box.reset_normals()
        self.assertEqual(box.f.shape, original_f.shape)
        for face, orig_face in zip(box.f, original_f):
            self.assertNotEqual(list(face), list(orig_face))
            self.assertEqual(set(face), set(orig_face))
        np.testing.assert_array_almost_equal(box.vn, np.negative(original_vn))

    def test_vert_connectivity(self):
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        connectivity = cube.vert_connectivity
        self.assertTrue(sp.issparse(connectivity))
        self.assertEqual(connectivity.shape, (cube.v.shape[0], cube.v.shape[0]))
        # Assert that neighbors are marked:
        for face in cube.f:
            face = np.asarray(face, dtype=np.uint32)
            self.assertNotEqual(connectivity[face[0], face[1]], 0)
            self.assertNotEqual(connectivity[face[1], face[2]], 0)
            self.assertNotEqual(connectivity[face[2], face[0]], 0)
        # Assert that non-neighbors are not marked:
        for v_index in set(cube.f.flatten()):
            faces_with_this_v = set(cube.f[np.any(cube.f == v_index, axis=1)].flatten())
            not_neighbors_of_this_v = set(cube.f.flatten()) - faces_with_this_v
            for vert in not_neighbors_of_this_v:
                self.assertEqual(connectivity[int(vert), int(v_index)], 0)
                self.assertEqual(connectivity[int(v_index), int(vert)], 0)

    def test_vert_opposites_per_edge(self):
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        opposites = cube.vert_opposites_per_edge
        self.assertIsInstance(opposites, dict)
        for e, op in opposites.items():
            self.assertIsInstance(e, tuple)
            self.assertEqual(len(e), 2)
            faces_with_e0 = set(cube.f[np.any(cube.f == e[0], axis=1)].flatten())
            faces_with_e1 = set(cube.f[np.any(cube.f == e[1], axis=1)].flatten())
            self.assertEqual(faces_with_e0.intersection(faces_with_e1) - set(e), set(op))

    def test_vertices_in_common(self):
        import timeit
        from lace.topology import vertices_in_common
        self.assertEqual(vertices_in_common([0, 1, 2], [0, 1, 3]), [0, 1])
        self.assertEqual(vertices_in_common([0, 1, 2], [3, 0, 1]), [0, 1])
        self.assertEqual(vertices_in_common([0, 1, 2], [3, 4, 5]), [])
        self.assertEqual(vertices_in_common([0, 1, 2], [0, 3, 4]), [0])
        self.assertEqual(vertices_in_common([0, 1, 2], [0, 1, 2]), [0, 1, 2])
        self.assertEqual(vertices_in_common([0, 1], [0, 1, 2]), [0, 1])
        self.assertEqual(vertices_in_common([0, 1, 2], [0, 1]), [0, 1])
        self.assertEqual(vertices_in_common([0, 1, 2], [0, 1, 2, 3]), [0, 1, 2])
        self.assertLess(timeit.timeit('vertices_in_common([0, 1, 2], [0, 1, 3])', setup='from lace.topology import vertices_in_common', number=10000), 0.015)

    def edges_the_hard_way(self, faces):
        from collections import Counter
        e0 = np.vstack((faces[:, 0], faces[:, 1]))
        e1 = np.vstack((faces[:, 1], faces[:, 2]))
        e2 = np.vstack((faces[:, 2], faces[:, 0]))
        e = np.hstack((e0, e1, e2)).T
        edge_count = Counter((min(a, b), max(a, b)) for a, b in e)
        return [x for x, count in edge_count.items() if count == 2]

    def test_faces_per_edge(self):
        import timeit
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        self.assertEqual(len(cube.faces_per_edge), len(self.edges_the_hard_way(cube.f)))
        for e in cube.faces_per_edge:
            # Check that each of these edges points to a pair of faces that
            # share two vertices -- that is, faces that share an edge.
            self.assertEqual(len(set(cube.f[e[0]]).intersection(set(cube.f[e[1]]))), 2)
        # Now check that changing the faces clears the cache
        cube.f = cube.f[[1, 2, 3, 4, 6, 7, 8, 9, 10, 11]] # remove [0 1 2] & [4 1 0] so edge [0, 1] is gone
        self.assertEqual(len(cube.faces_per_edge), len(self.edges_the_hard_way(cube.f)))
        for e in cube.faces_per_edge:
            self.assertEqual(len(set(cube.f[e[0]]).intersection(set(cube.f[e[1]]))), 2)
        # And test that caching happens -- without caching, this takes about 5 seconds:
        self.assertLess(timeit.timeit('cube.faces_per_edge', setup='from lace.shapes import create_cube; cube = create_cube([0., 0., 0.], 1.)', number=10000), 0.01)

    def test_vertices_per_edge(self):
        import timeit
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        self.assertEqual(len(cube.vertices_per_edge), len(self.edges_the_hard_way(cube.f)))
        self.assertEqual(set([(min(a, b), max(a, b)) for a, b in cube.vertices_per_edge]), set(self.edges_the_hard_way(cube.f)))
        # Now check that changing the faces clears the cache
        cube.f = cube.f[[1, 2, 3, 4, 6, 7, 8, 9, 10, 11]] # remove [0 1 2] & [4 1 0] so edge [0, 1] is gone
        self.assertEqual(len(cube.vertices_per_edge), len(self.edges_the_hard_way(cube.f)))
        self.assertEqual(set([(min(a, b), max(a, b)) for a, b in cube.vertices_per_edge]), set(self.edges_the_hard_way(cube.f)))
        # And test that caching happens -- without caching, this takes about 5 seconds:
        self.assertLess(timeit.timeit('cube.vertices_per_edge', setup='from lace.shapes import create_cube; cube = create_cube([0., 0., 0.], 1.)', number=10000), 0.01)

    def test_vertices_to_edges_matrix(self):
        import timeit
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        calculated_edges = cube.vertices_to_edges_matrix.dot(cube.v.ravel()).reshape((-1, 3))
        self.assertEqual(len(calculated_edges), len(cube.vertices_per_edge))
        for e, e_ind in zip(calculated_edges, cube.vertices_per_edge):
            np.testing.assert_array_equal(e, cube.v[e_ind[0]] - cube.v[e_ind[1]])
        # And test that caching happens -- without caching, this takes about 5 seconds:
        self.assertLess(timeit.timeit('cube.vertices_to_edges_matrix', setup='from lace.shapes import create_cube; cube = create_cube([0., 0., 0.], 1.)', number=10000), 0.01)

    def vertices_to_edges_matrix_single_axis(self):
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        # Assert that it produces the same results as vertices_to_edges_matrix:
        self.assertEqual(np.vstack((cube.vertices_to_edges_matrix_single_axis.dot(cube.v[:, ii]) for ii in range(3))).T,
                         cube.vertices_to_edges_matrix.dot(cube.v.ravel()).reshape((-1, 3)))

    def test_remove_redundant_verts(self):
        eps = 1e-15
        from lace.shapes import create_cube
        cube = create_cube(np.zeros(3), 1.)
        orig_v = cube.v.copy()
        orig_f = cube.f.copy()
        cube.f[1:4] = cube.f[1:4] + len(cube.v)
        cube.v = np.vstack((cube.v, cube.v + eps))
        cube.remove_redundant_verts()
        np.testing.assert_array_equal(cube.v, orig_v)
        np.testing.assert_array_equal(cube.f, orig_f)

    def test_has_same_topology(self):
        from lace.shapes import create_cube

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        self.assertTrue(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.ones(3), 1.)
        self.assertTrue(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        cube_2.f = np.roll(cube_2.f, 1, axis=1)
        self.assertFalse(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        del cube_2.f
        self.assertFalse(cube_1.has_same_topology(cube_2))
