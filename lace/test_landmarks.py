import unittest
import numpy as np
from bltest import skip_on_import_error
from bltest.extra_asserts import ExtraAssertionsMixin
from lace.cache import sc
from lace.mesh import Mesh
from .testing.scratch_dir import ScratchDirMixin

class TestLandmarks(ExtraAssertionsMixin, ScratchDirMixin, unittest.TestCase):
    def setUp(self):
        skip_on_import_error('lace-search')
        self.scan_fname = sc('s3://bodylabs-korper-assets/is/ps/shared/data/body/caesar/RawScans/csr0001a.ply')
        self.scan_lmrk = sc('s3://bodylabs-korper-assets/is/ps/shared/data/body/caesar/Landmarks/csr0001a.lmrk')
        self.template_fname = sc('s3://bodylabs-korper-assets/is/ps/shared/data/body/template/textured_mean_scape_female.obj')
        self.template_pp = sc('s3://bodylabs-korper-assets/is/ps/shared/data/body/template/template_caesar_picked_points.pp')
        self.scan = Mesh(filename=self.scan_fname, lmrkfilename=self.scan_lmrk)
        self.template = Mesh(filename=self.template_fname, ppfilename=self.template_pp)
        self.template_without_regressors = Mesh(filename=self.template_fname, ppfilename=self.template_pp)
        self.template_without_regressors.landm_regressors = {}
        super(TestLandmarks, self).setUp()

    def test_CAESAR_lmrk_file(self):
        m = Mesh(filename=self.scan_fname, landmarks=self.scan_lmrk)
        self.assertEqual(m.landm, self.scan.landm)
        self.assertDictOfArraysEqual(m.landm_xyz, self.scan.landm_xyz)

    def test_meshlab_pp_file(self):
        m = Mesh(filename=self.template_fname, landmarks=self.template_pp)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def make_index_data(self, loader, extension):
        import os
        path = os.path.join(self.scratch_dir, 'landmarks' + extension)
        test_data = {name: long(val) for name, val in self.template.landm.items()}
        loader.dump(test_data, path)
        return path

    def make_point_data(self, loader, extension):
        import os
        path = os.path.join(self.scratch_dir, 'landmarks' + extension)
        test_data = {name: val.tolist() for name, val in self.template.landm_xyz.items()}
        loader.dump(test_data, path)
        return path

    def test_yaml_index(self):
        from baiji.serialization import yaml
        path = self.make_index_data(yaml, '.yaml')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template_without_regressors.landm_xyz)

    def test_yaml_points(self):
        from baiji.serialization import yaml
        path = self.make_point_data(yaml, '.yaml')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def test_yml_index(self):
        from baiji.serialization import yaml
        path = self.make_index_data(yaml, '.yml')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template_without_regressors.landm_xyz)

    def test_yml_points(self):
        from baiji.serialization import yaml
        path = self.make_point_data(yaml, '.yml')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def test_json_index(self):
        from baiji.serialization import json
        path = self.make_index_data(json, '.json')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template_without_regressors.landm_xyz)

    def test_json_points(self):
        from baiji.serialization import json
        path = self.make_point_data(json, '.json')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def test_pickle_index(self):
        from baiji.serialization import pickle
        path = self.make_index_data(pickle, '.pkl')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template_without_regressors.landm_xyz)

    def test_pickle_points(self):
        from baiji.serialization import pickle
        path = self.make_point_data(pickle, '.pkl')
        m = Mesh(filename=self.template_fname, landmarks=path)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def test_explicit_landmark_indexes_from_another_mesh(self):
        m = Mesh(filename=self.template_fname, landmarks=self.template.landm)
        np.testing.assert_array_equal(self.template.v, m.v)
        self.assertEqual(m.landm, self.template.landm)
        # The following are _not_ expected to pass, as the initial load of template is
        # loading points that are slightly off from the actual vertex locations,
        # so the regressors are going to be subtly different.
        #   self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)
        #   self.assertDictOfArraysAlmostEqual(m.landm_regressors, self.template.landm_regressors)
        # But if we disable the regressors and just use the landmark verts:
        m.landm_regressors = {}
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template_without_regressors.landm_xyz)

    def test_explicit_landmark_points_from_another_mesh(self):
        m = Mesh(filename=self.template_fname, landmarks=self.template.landm_xyz)
        self.assertEqual(m.landm, self.template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, self.template.landm_xyz)

    def test_explicit_landmarks_as_list_of_indexes(self):
        m = Mesh(filename=self.template_fname, landmarks=[0, 7])
        self.assertEqual(m.landm, {'0': 0, '1': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': self.template.v[0], '1': self.template.v[7]})

    def test_explicit_landmarks_as_array_of_indexes(self):
        m = Mesh(filename=self.template_fname, landmarks=np.array([0, 7]))
        self.assertEqual(m.landm, {'0': 0, '1': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': self.template.v[0], '1': self.template.v[7]})

    def test_explicit_landmarks_as_dict_of_indexes(self):
        m = Mesh(filename=self.template_fname, landmarks={'foo': 0, 'bar': 7})
        self.assertEqual(m.landm, {'foo': 0, 'bar': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'foo': self.template.v[0], 'bar': self.template.v[7]})

    def test_explicit_landmarks_as_array_of_points(self):
        m = Mesh(filename=self.template_fname, landmarks=[self.template.v[0], self.template.v[7]])
        self.assertEqual(m.landm, {'0': 0, '1': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': self.template.v[0], '1': self.template.v[7]})

    def test_explicit_landmarks_as_list_of_points(self):
        m = Mesh(filename=self.template_fname, landmarks=[self.template.v[0].tolist(), self.template.v[7].tolist()])
        self.assertEqual(m.landm, {'0': 0, '1': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': self.template.v[0], '1': self.template.v[7]})

    def test_explicit_landmarks_as_dict_of_points(self):
        m = Mesh(filename=self.template_fname, landmarks={'foo': self.template.v[0], 'bar': self.template.v[7]})
        self.assertEqual(m.landm, {'foo': 0, 'bar': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'foo': self.template.v[0], 'bar': self.template.v[7]})

    def test_explicit_landmarks_as_array_of_close_points(self):
        m = Mesh(filename=self.template_fname, landmarks=[self.template.v[0]+.000001, self.template.v[7]+.000001])
        self.assertEqual(m.landm, {'0': 0, '1': 7})
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': self.template.v[0], '1': self.template.v[7]})
