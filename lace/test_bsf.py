import unittest
import numpy as np
from lace.serialization import bsf, ply
from lace.cache import vc

class TestBSF(unittest.TestCase):

    def test_load_bsf(self):
        expected_mesh = ply.load(vc('/unittest/bsf/bsf_example.ply'))
        bsf_mesh = bsf.load(vc('/unittest/bsf/bsf_example.bsf'))
        np.testing.assert_array_almost_equal(bsf_mesh.v, expected_mesh.v)
        np.testing.assert_equal(bsf_mesh.f, expected_mesh.f)
