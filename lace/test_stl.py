import unittest
import numpy as np
from lace.cache import sc
from lace.serialization import stl

class TestSTL(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp('bodylabs-test')
        self.truth = {
            'box_v': np.array([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]]).T,
            'box_f': np.array([[0, 1, 2], [3, 2, 1], [0, 2, 4], [6, 4, 2], [0, 4, 1], [5, 1, 4], [7, 5, 6], [4, 6, 5], [7, 6, 3], [2, 3, 6], [7, 3, 5], [1, 5, 3]]),
            'box_fn': np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, -1], [-0, -0, -1], [0, -1, 0], [-0, -1, -0], [-1, 0, 0], [-1, -0, -0]]),
        }
        # Because STL gives duplicated verts
        self.truth['box_v'] = self.truth['box_v'][self.truth['box_f'].flatten()]
        self.truth['box_f'] = np.array(range(self.truth['box_v'].shape[0])).reshape((-1, 3))
        self.test_stl_url = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_box.stl"
        self.test_stl_path = sc(self.test_stl_url)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_loads_from_local_path_using_serializer(self):
        m = stl.load(self.test_stl_path)
        np.testing.assert_array_equal(m.v, self.truth['box_v'])
        np.testing.assert_array_equal(m.f, self.truth['box_f'])
        np.testing.assert_array_equal(m.fn, self.truth['box_fn'])
