import unittest
import numpy as np
from lace.cache import sc
from lace.mesh import Mesh

class TestGeometryMixin(unittest.TestCase):
    debug = False

    def test_cut_across_axis(self):
        original_mesh = Mesh(filename=sc('s3://bodylabs-assets/example_meshes/average_female.obj'))

        # Set up
        mesh = original_mesh.copy()
        # Sanity check
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.3668)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.673871)
        # Act
        mesh.cut_across_axis(0, minval=-0.1)
        # Test
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.1, decimal=1)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.673871)
        # Visualize
        if self.debug:
            mesh.show()

        # Set up
        mesh = original_mesh.copy()
        # Sanity check
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.3668)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.673871)
        # Act
        mesh.cut_across_axis(0, maxval=0.1)
        # Test
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.3668)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.1, decimal=1)
        # Visualize
        if self.debug:
            mesh.show()

        # Set up
        mesh = original_mesh.copy()
        # Sanity check
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.3668)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.673871)
        # Act
        mesh.cut_across_axis(0, minval=-0.1, maxval=0.1)
        # Test
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.1, decimal=1)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.1, decimal=1)
        # Visualize
        if self.debug:
            mesh.show()

    def test_cut_across_axis_by_percentile(self):
        original_mesh = Mesh(filename=sc('s3://bodylabs-assets/example_meshes/average_female.obj'))

        # Set up
        mesh = original_mesh.copy()
        # Sanity check
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), -0.3668)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.673871)
        # Act
        mesh.cut_across_axis_by_percentile(0, 25, 40)
        # Test
        np.testing.assert_almost_equal(mesh.v[:, 0].min(), 0.03, decimal=1)
        np.testing.assert_almost_equal(mesh.v[:, 0].max(), 0.10, decimal=1)
        # Visualize
        if self.debug:
            mesh.show()

    def test_first_blip(self):
        '''
        Create up a triagonal prism, the base of which is an equilateral
        triangle with one side along the x axis and its third point on the
        +y axis.

        Take three origins and vectors whose first_blip should be the same
        vertex.

        '''
        import math
        from lace import shapes

        points = np.array([
            [1, 0, 0],
            [0, math.sqrt(1.25), 0],
            [-1, 0, 0],
        ])
        prism = shapes.create_triangular_prism(*points, height=4)

        test_cases = [
            {'origin': [0, -0.5, 0], 'initial_direction': [-1, 0.3, 0]},
            {'origin': [-1.1, -0.5, 0], 'initial_direction': [0, 1, 0]},
            {'origin': [-1, 1, 0], 'initial_direction': [-1, -1, 0]},
        ]

        # TODO This is a little sloppy. Because flatten_dim=2,
        # [-1, 0, 0] and [-1, 0, 4] are equally valid results.
        expected_v = [-1, 0, 0]

        for test_case in test_cases:
            np.testing.assert_array_equal(
                prism.first_blip(2, test_case['origin'], test_case['initial_direction']),
                expected_v
            )

    def test_first_blip_ignores_squash_axis(self):
        '''
        The above test is nice, but since the object is a prism, it's the same along
        the z-axis -- and hence isn't really testing squash_axis. Here's another test
        with a different setup which tries to get at it.

        '''
        import math
        from lace import shapes

        points = np.array([
            [1, 0, 0],
            [0, math.sqrt(1.25), 0],
            [-1, 0, 0],
        ])
        prism = shapes.create_triangular_prism(*points, height=4)

        test_cases = [
            {'origin': [-1, 0, -1], 'initial_direction': [0, 1, 0]},
        ]

        expected_v = [0, math.sqrt(1.25), 0]

        for test_case in test_cases:
            np.testing.assert_array_equal(
                prism.first_blip(0, test_case['origin'], test_case['initial_direction']),
                expected_v
            )

    def test_reorient_is_noop_on_empty_mesh(self):
        mesh = Mesh()
        mesh.reorient(up=[0, 1, 0], look=[0, 0, 1])

    def test_scale_is_noop_on_empty_mesh(self):
        mesh = Mesh()
        mesh.scale(7)

    def test_translate_is_noop_on_empty_mesh(self):
        mesh = Mesh()
        mesh.translate(1)

    def test_centroid_is_undefined_on_empty_mesh(self):
        mesh = Mesh()

        with self.assertRaises(ValueError) as ctx:
            mesh.centroid # pylint: disable=pointless-statement

        self.assertEqual(
            str(ctx.exception),
            'Mesh has no vertices; centroid is not defined'
        )

    def test_bounding_box_is_undefined_on_empty_mesh(self):
        mesh = Mesh()

        with self.assertRaises(ValueError) as ctx:
            mesh.bounding_box # pylint: disable=pointless-statement

        self.assertEqual(
            str(ctx.exception),
            'Mesh has no vertices; bounding box is not defined'
        )

    def test_recenter_over_floor_raises_expected_on_empty_mesh(self):
        mesh = Mesh()

        with self.assertRaises(ValueError) as ctx:
            mesh.recenter_over_floor()

        self.assertEqual(
            str(ctx.exception),
            'Mesh has no vertices; centroid is not defined'
        )

    def test_predict_body_units_m(self):
        mesh = Mesh()
        mesh.v = np.array([[1.5, 1.0, 0.5], [-1.5, 0.5, 1.5]])
        self.assertEqual(mesh.predict_body_units(), 'm')

    def test_predict_body_units_cm(self):
        mesh = Mesh()
        mesh.v = np.array([[150, 100, 50], [-150, 50, 150]])
        self.assertEqual(mesh.predict_body_units(), 'cm')
