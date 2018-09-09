import unittest
import numpy as np
from bltest import attr
from lace.mesh import Mesh
from lace.cache import sc


class TestMesh(unittest.TestCase):
    def test_v_converts_datatype(self):
        m = Mesh()
        m.v = np.array([[1, 1, 1]], dtype=np.uint32)
        self.assertEqual(m.v.dtype, np.float64)

    def test_v_accepts_n_by_3(self):
        a = np.zeros((10, 3))
        m = Mesh(v=a)
        np.testing.assert_array_equal(m.v, a)

    def test_v_transposes_3_by_n(self):
        a = np.zeros((3, 10))
        m = Mesh(v=a)
        np.testing.assert_array_equal(m.v, a.T)

    def test_v_rejects_without_one_dim_being_3(self):
        a = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            Mesh(v=a)

    def test_v_converts_to_np_array(self):
        a = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        m = Mesh(v=a)
        np.testing.assert_array_equal(m.v, a)
        self.assertIsInstance(m.v, np.ndarray)

    def test_f_converts_datatype(self):
        m = Mesh()
        m.f = np.array([[1, 1, 1]], dtype=np.float64)
        self.assertEqual(m.f.dtype, np.uint64)

    def test_vc_accepts_n_by_3(self):
        a = np.zeros((10, 3))
        m = Mesh(v=a, vc=a)
        np.testing.assert_array_equal(m.vc, a)

    def test_fc_accepts_n_by_3(self):
        a = np.zeros((10, 3))
        m = Mesh(f=a, fc=a)
        np.testing.assert_array_equal(m.fc, a)

    def test_vn_accepts_n_by_3(self):
        a = np.zeros((10, 3))
        m = Mesh(vn=a)
        np.testing.assert_array_equal(m.vn, a)

    def test_fn_accepts_n_by_3(self):
        a = np.zeros((10, 3))
        m = Mesh(fn=a)
        np.testing.assert_array_equal(m.fn, a)

    @attr('missing_assets')
    def test_estimate_vertex_normals(self):
        # normals of a sphere should be scaled versions of the vertices
        test_sphere_path = sc(
            's3://bodylabs-korper-assets/is/ps/shared/data/body/'
            'korper_testdata/sphere.ply'
        )
        m = Mesh(filename=test_sphere_path)
        m.v -= np.mean(m.v, axis=0)
        rad = np.linalg.norm(m.v[0])
        m.estimate_vertex_normals()
        mse = np.mean(np.sqrt(np.sum((m.vn - m.v/rad)**2, axis=1)))
        self.assertTrue(mse < 0.05)

    def test_mesh_from_mesh(self):
        m = Mesh(
            v=np.random.randn(10, 3),
            f=np.random.randn(10, 3),
            vc=np.random.randn(10, 3),
            vn=np.random.randn(10, 3),
        )
        m2 = Mesh(m)
        np.testing.assert_array_equal(m.v, m2.v)
        np.testing.assert_array_equal(m.f, m2.f)
        np.testing.assert_array_equal(m.vc, m2.vc)
        np.testing.assert_array_equal(m.vn, m2.vn)
        self.assertNotEqual(id(m.v), id(m2.v))

    def test_mesh_from_v(self):
        a = np.zeros((10, 3))
        m = Mesh(a)
        np.testing.assert_array_equal(m.v, a)

    def test_mesh_copy(self):
        m = Mesh(
            v=np.random.randn(10, 3),
            f=np.random.randn(10, 3),
            vc=np.random.randn(10, 3),
            vn=np.random.randn(10, 3),
        )
        m2 = m.copy()
        np.testing.assert_array_equal(m.v, m2.v)
        np.testing.assert_array_equal(m.f, m2.f)
        np.testing.assert_array_equal(m.vc, m2.vc)
        np.testing.assert_array_equal(m.vn, m2.vn)
        self.assertNotEqual(id(m.v), id(m2.v))

    def test_mesh_copy_light(self):
        m = Mesh(
            v=np.random.randn(10, 3),
            f=np.random.randn(10, 3),
            vc=np.random.randn(10, 3),
            vn=np.random.randn(10, 3),
        )
        m2 = m.copy(only=['f', 'v'])
        np.testing.assert_array_equal(m.v, m2.v)
        np.testing.assert_array_equal(m.f, m2.f)
        self.assertIsNone(m2.vc)
        self.assertIsNone(m2.vn)
        self.assertNotEqual(id(m.v), id(m2.v))

    def test_copy_mesh(self):
        import copy
        m = Mesh(
            v=np.random.randn(10, 3),
            f=np.random.randn(10, 3),
            vc=np.random.randn(10, 3),
            vn=np.random.randn(10, 3),
        )
        m2 = copy.copy(m)
        np.testing.assert_array_equal(m.v, m2.v)
        np.testing.assert_array_equal(m.f, m2.f)
        np.testing.assert_array_equal(m.vc, m2.vc)
        np.testing.assert_array_equal(m.vn, m2.vn)
        self.assertEqual(id(m.v), id(m2.v))

    def test_deepcopy_mesh(self):
        import copy
        m = Mesh(
            v=np.random.randn(10, 3),
            f=np.random.randn(10, 3),
            vc=np.random.randn(10, 3),
            vn=np.random.randn(10, 3),
        )
        m2 = copy.deepcopy(m)
        np.testing.assert_array_equal(m.v, m2.v)
        np.testing.assert_array_equal(m.f, m2.f)
        np.testing.assert_array_equal(m.vc, m2.vc)
        np.testing.assert_array_equal(m.vn, m2.vn)
        self.assertNotEqual(id(m.v), id(m2.v))

    def test_concatenate(self):
        v1 = np.array([
            [0., 0., 0.],
            [5., 5., 5.],
            [5., 0., 5.],
        ])
        vc1 = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        f1 = np.array([
            [0, 1, 2],
        ])
        m1 = Mesh(v=v1, vc=vc1, f=f1)

        v2 = np.array([
            [0., 0., 0.],
            [5., -5., 5.],
            [5., 0., 5.],
        ])
        vc2 = np.array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.],
        ])
        f2 = np.array([
            [1, 0, 2],
        ])
        m2 = Mesh(v=v2, vc=vc2, f=f2)

        v3 = np.array([
            [0., 0., 0.],
            [3., -3., 3.],
            [3., 0., 3.],
            # Add an extra unused vertex to make the counts a little more
            # interesting.
            [3., 1., 7.],
        ])
        vc3 = np.array([
            [0., 0.5, 1.],
            [0.5, 0., 1.],
            [1., 1., 0.],
            [1., 0.5, 1.],
        ])
        f3 = np.array([
            [2, 1, 0],
        ])
        m3 = Mesh(v=v3, vc=vc3, f=f3)

        mcat = Mesh.concatenate(m1, m2, m3)

        num_v1 = v1.shape[0]
        num_v2 = v2.shape[0]
        num_v3 = v3.shape[0]
        num_vertices = num_v1 + num_v2 + num_v3

        # Check expected number of vertices, colors, and faces.
        self.assertEqual(mcat.v.shape[0], num_vertices)
        self.assertEqual(mcat.vc.shape[0], num_vertices)
        self.assertEqual(mcat.f.shape[0], 3)

        # Check the vertices.
        np.testing.assert_array_equal(v1, mcat.v[:num_v1, :])
        np.testing.assert_array_equal(v2, mcat.v[num_v1:(num_v1 + num_v2), :])
        np.testing.assert_array_equal(v3, mcat.v[(num_v1 + num_v2):, :])
        # Check vertex colors
        np.testing.assert_array_equal(vc1, mcat.vc[:3, :])
        np.testing.assert_array_equal(vc2, mcat.vc[3:6, :])
        np.testing.assert_array_equal(vc3, mcat.vc[6:, :])
        # Check faces.
        np.testing.assert_array_equal(
            f1.ravel(), mcat.f[0, :].ravel())
        np.testing.assert_array_equal(
            f2.ravel() + num_v1, mcat.f[1, :].ravel())
        np.testing.assert_array_equal(
            f3.ravel() + num_v1 + num_v2, mcat.f[2, :].ravel())

    def test_concatenate_partial_missing_vc(self):
        v1 = np.array([
            [0., 0., 0.],
            [5., 5., 5.],
            [5., 0., 5.],
        ])
        vc1 = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        m1 = Mesh(v=v1, vc=vc1)

        v2 = np.array([
            [0., 0., 0.],
            [5., -5., 5.],
            [5., 0., 5.],
        ])
        m2 = Mesh(v=v2)

        with self.assertRaisesRegexp(ValueError, 'all args or none.'):
            Mesh.concatenate(m1, m2)

    def test_concatenate_vertices_only(self):
        v1 = np.array([
            [0., 0., 0.],
            [5., 5., 5.],
            [5., 0., 5.],
        ])
        m1 = Mesh(v=v1)

        v2 = np.array([
            [0., 0., 0.],
            [5., -5., 5.],
            [5., 0., 5.],
        ])
        m2 = Mesh(v=v2)

        self.assertEqual(Mesh.concatenate(m1, m2).v.shape[0], 6)
