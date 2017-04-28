# pylint: disable=len-as-condition
import unittest
import numpy as np
import mock
from lace.mesh import Mesh

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
        np.testing.assert_array_equal(quads_to_tris(tris), expected_quads)

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

    def test_keep_vertices(self):
        from lace.cache import sc
        from lace.serialization import obj
        mesh = obj.load(sc('s3://bodylabs-versioned-assets/templates/cached_model_templates/sm_2013_f_0005.1.0.0.obj'))

        indices_to_keep, expected_verts, expected_face_vertices = self.indicies_for_testing_keep_vertices(mesh)

        mesh.keep_vertices(indices_to_keep)

        np.testing.assert_array_equal(mesh.v, expected_verts)

        np.testing.assert_array_equal(mesh.v[mesh.f.flatten()], expected_face_vertices)

        max_v_index = np.max(mesh.f.flatten())
        self.assertLessEqual(max_v_index, mesh.v.shape[0] - 1)

    def test_keep_vertices_without_segm(self):
        from lace.cache import sc
        from lace.serialization import obj
        mesh = obj.load(sc('s3://bodylabs-versioned-assets/templates/cached_model_templates/sm_2013_f_0005.1.0.0.obj'))
        mesh.segm = None

        indices_to_keep, expected_verts, expected_face_vertices = self.indicies_for_testing_keep_vertices(mesh)

        mesh.keep_vertices(indices_to_keep)

        np.testing.assert_array_equal(mesh.v, expected_verts)

        np.testing.assert_array_equal(mesh.v[mesh.f.flatten()], expected_face_vertices)

        max_v_index = np.max(mesh.f.flatten())
        self.assertLessEqual(max_v_index, mesh.v.shape[0] - 1)

    def test_keep_vertices_without_f(self):
        from lace.cache import sc
        from lace.serialization import obj
        mesh = obj.load(sc('s3://bodylabs-versioned-assets/templates/cached_model_templates/sm_2013_f_0005.1.0.0.obj'))
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

    @mock.patch('warnings.warn')
    def test_keep_vertices_with_empty_list_does_not_warn(self, warn):
        from lace.cache import sc
        from lace.serialization import obj
        mesh = obj.load(sc('s3://bodylabs-versioned-assets/templates/cached_model_templates/sm_2013_f_0005.1.0.0.obj'))

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

    def test_keep_segments(self):
        from lace.cache import sc
        from lace.serialization import obj
        mesh = obj.load(sc('s3://bodylabs-versioned-assets/templates/cached_model_templates/sm_2013_f_0005.1.0.0.obj'))

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
