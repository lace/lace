import unittest
import numpy as np
from lace.mesh import Mesh
from lace import color

class TestMeshColors(unittest.TestCase):
    def setUp(self):
        self.colormap = 'jet'
        self.jet_0 = np.array([0.0, 0.0, 0.5])
        self.jet_1 = np.array([0.5, 0.0, 0.0])

    def test_colors_like_with_full_array(self):
        c = np.random.rand(10, 3)
        v = np.ones((10, 3))
        vc = color.colors_like(c, v)
        np.testing.assert_array_equal(c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_full_array_transposed(self):
        c = np.random.rand(10, 3)
        v = np.ones((10, 3))
        vc = color.colors_like(c.T, v)
        np.testing.assert_array_equal(c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_single_row(self):
        c = np.array([[1.0, 0.0, 1.0, 0.0]])
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([self.jet_1, self.jet_0, self.jet_1, self.jet_0])
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_single_col(self):
        c = np.array([[1.0], [0.0], [1.0], [0.0]])
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([self.jet_1, self.jet_0, self.jet_1, self.jet_0])
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_array(self):
        c = np.array([1.0, 0.0, 1.0, 0.0])
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([self.jet_1, self.jet_0, self.jet_1, self.jet_0])
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_list(self):
        c = [1.0, 0.0, 1.0, 0.0]
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([self.jet_1, self.jet_0, self.jet_1, self.jet_0])
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_with_name(self):
        c = 'red'
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([1.0, 0.0, 0.0]) * v
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_rgb_triple_as_array(self):
        c = [1.0, 0.0, 1.0]
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([1.0, 0.0, 1.0]) * v
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_rgb_triple_as_col(self):
        c = [[1.0], [0.0], [1.0]]
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([1.0, 0.0, 1.0]) * v
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_rgb_triple_as_row(self):
        c = [[1.0, 0.0, 1.0]]
        v = np.ones((4, 3))
        vc = color.colors_like(c, v, colormap=self.colormap)
        expected_c = np.array([1.0, 0.0, 1.0]) * v
        np.testing.assert_array_equal(expected_c, vc)
        self.assertEqual(vc.dtype, np.float64)

    def test_colors_like_none_or_empty(self):
        for c in [None, [], (), np.array([])]:
            v = np.ones((4, 3))
            vc = color.colors_like(c, v, colormap=self.colormap)
            self.assertIsNone(vc)

    def test_setting_vc_in_mesh_constructor(self):
        m = Mesh(v=np.ones((4, 3)), vc='red')
        expected_c = np.array([1.0, 0.0, 0.0]) * m.v
        np.testing.assert_array_equal(expected_c, m.vc)
        self.assertEqual(m.vc.dtype, np.float64)

    def test_scale_vertex_colors(self):
        m = Mesh(v=np.ones((4, 3)), vc='red')
        m.scale_vertex_colors(np.array([2.0, 1.0, 0.5, 0.0]))
        expected_c = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(expected_c, m.vc)
        self.assertEqual(m.vc.dtype, np.float64)
