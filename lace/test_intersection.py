import unittest
import numpy as np
from bltest import attr
from lace.mesh import Mesh
from polliwog import Plane, Polyline
from collections import namedtuple

class TestIntersection(unittest.TestCase):
    def setUp(self):
        self.box_mesh = Mesh(v=np.array([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]]).T,
                                 f=np.array([[0, 1, 2], [3, 2, 1], [0, 2, 4], [6, 4, 2], [0, 4, 1], [5, 1, 4], [7, 5, 6], [4, 6, 5], [7, 6, 3], [2, 3, 6], [7, 3, 5], [1, 5, 3]]))

    def double_mesh(self, mesh, shift=[2., 0., 0.]):
        other_mesh = Mesh(v=mesh.v + np.array(shift), f=mesh.f)
        two_meshes = Mesh(v=np.vstack((mesh.v, other_mesh.v)),
                              f=np.vstack((mesh.f, other_mesh.f + mesh.v.shape[0])))
        return two_meshes

    def test_mesh_plane_intersection(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        # Verify that we're finding the correct number of faces to start with
        self.assertEqual(len(self.box_mesh.faces_intersecting_plane(plane)), 8)

        xsections = self.box_mesh.intersect_plane(plane)
        self.assertIsInstance(xsections, list)
        self.assertEqual(len(xsections), 1)
        self.assertEqual(len(xsections[0].v), 8)
        self.assertTrue(xsections[0].is_closed)

        self.assertEqual(xsections[0].total_length, 4.0)
        np.testing.assert_array_equal(xsections[0].v[:, 1], np.zeros((8, )))
        for a, b in zip(xsections[0].v[0:-1, [0, 2]], xsections[0].v[1:, [0, 2]]):
            # Each line changes only one coordinate, and is 0.5 long
            self.assertEqual(np.sum(a == b), 1)
            self.assertEqual(np.linalg.norm(a - b), 0.5)

    def test_mesh_plane_intersection_with_ret_pointcloud(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])
        plane = Plane(sample, normal)

        xsections = self.box_mesh.intersect_plane(plane)

        pointcloud = self.box_mesh.intersect_plane(plane, ret_pointcloud=True)
        self.assertIsInstance(pointcloud, np.ndarray)
        np.testing.assert_array_equal(pointcloud, xsections[0].v)

    def test_mesh_plane_intersection_with_no_intersection(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 5., 0.])

        plane = Plane(sample, normal)

        # Verify that we're detecting faces that lay entirely in the plane as potential intersections
        self.assertEqual(len(self.box_mesh.faces_intersecting_plane(plane)), 0)

        xsections = self.box_mesh.intersect_plane(plane)
        self.assertIsInstance(xsections, list)
        self.assertEqual(len(xsections), 0)

    def test_mesh_plane_intersection_with_no_intersection_and_ret_pointcloud(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 5., 0.])

        plane = Plane(sample, normal)

        xsections = self.box_mesh.intersect_plane(plane)

        pointcloud = self.box_mesh.intersect_plane(plane, ret_pointcloud=True)

        self.assertIsInstance(pointcloud, np.ndarray)
        assert pointcloud.shape == (0,3)

    def test_mesh_plane_intersection_wth_two_components(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        two_box_mesh = self.double_mesh(self.box_mesh)

        xsections = two_box_mesh.intersect_plane(plane)
        self.assertIsInstance(xsections, list)
        self.assertEqual(len(xsections), 2)
        self.assertEqual(len(xsections[0].v), 8)
        self.assertTrue(xsections[0].is_closed)
        self.assertEqual(len(xsections[1].v), 8)
        self.assertTrue(xsections[1].is_closed)

    def test_mesh_plane_intersection_wth_neighborhood(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        two_box_mesh = self.double_mesh(self.box_mesh)

        xsection = two_box_mesh.intersect_plane(plane, neighborhood=np.array([[0., 0., 0.]]))
        self.assertIsInstance(xsection, Polyline)
        self.assertEqual(len(xsection.v), 8)
        self.assertTrue(xsection.is_closed)

    def test_mesh_plane_intersection_with_neighborhood_and_ret_pointcloud(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])
        plane = Plane(sample, normal)

        two_box_mesh = self.double_mesh(self.box_mesh)

        neighborhood=np.array([[0.0, 0.0, 0.0]])
        xsection = two_box_mesh.intersect_plane(plane, neighborhood=neighborhood)

        pointcloud = two_box_mesh.intersect_plane(plane, neighborhood=neighborhood, ret_pointcloud=True)

        self.assertIsInstance(pointcloud, np.ndarray)
        np.testing.assert_array_equal(pointcloud, xsection.v)

    def test_mesh_plane_intersection_with_non_watertight_mesh(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        v_on_faces_to_remove = np.nonzero(self.box_mesh.v[:, 0] < 0.0)[0]
        faces_to_remove = np.all(np.in1d(self.box_mesh.f.ravel(), v_on_faces_to_remove).reshape((-1, 3)), axis=1)
        open_mesh = Mesh(v=self.box_mesh.v, f=self.box_mesh.f[np.logical_not(faces_to_remove)])

        xsections = open_mesh.intersect_plane(plane)

        # The removed side is not in the xsection:
        self.assertFalse(any(np.all(xsections[0].v == [-0.5, 0., 0.], axis=1)))

        self.assertIsInstance(xsections, list)
        self.assertEqual(len(xsections), 1)
        self.assertEqual(len(xsections[0].v), 7)
        self.assertFalse(xsections[0].is_closed)

        self.assertEqual(xsections[0].total_length, 3.0)
        np.testing.assert_array_equal(xsections[0].v[:, 1], np.zeros((7, )))
        for a, b in zip(xsections[0].v[0:-1, [0, 2]], xsections[0].v[1:, [0, 2]]):
            # Each line changes only one coordinate, and is 0.5 long
            self.assertEqual(np.sum(a == b), 1)
            self.assertEqual(np.linalg.norm(a - b), 0.5)

    def test_mesh_plane_intersection_with_mulitple_non_watertight_meshes(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        v_on_faces_to_remove = np.nonzero(self.box_mesh.v[:, 0] < 0.0)[0]
        faces_to_remove = np.all(np.in1d(self.box_mesh.f.ravel(), v_on_faces_to_remove).reshape((-1, 3)), axis=1)
        open_mesh = Mesh(v=self.box_mesh.v, f=self.box_mesh.f[np.logical_not(faces_to_remove)])
        two_open_meshes = self.double_mesh(open_mesh)

        xsections = two_open_meshes.intersect_plane(plane)

        # The removed side is not in the xsection:
        self.assertFalse(any(np.all(xsections[0].v == [-0.5, 0., 0.], axis=1)))

        self.assertIsInstance(xsections, list)
        self.assertEqual(len(xsections), 2)
        self.assertEqual(len(xsections[0].v), 7)
        self.assertFalse(xsections[0].is_closed)
        self.assertEqual(len(xsections[1].v), 7)
        self.assertFalse(xsections[1].is_closed)

        self.assertEqual(xsections[0].total_length, 3.0)
        np.testing.assert_array_equal(xsections[0].v[:, 1], np.zeros((7, )))
        np.testing.assert_array_equal(xsections[1].v[:, 1], np.zeros((7, )))
        for a, b in zip(xsections[0].v[0:-1, [0, 2]], xsections[0].v[1:, [0, 2]]):
            # Each line changes only one coordinate, and is 0.5 long
            self.assertEqual(np.sum(a == b), 1)
            self.assertEqual(np.linalg.norm(a - b), 0.5)
