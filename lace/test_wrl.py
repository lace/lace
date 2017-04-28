import unittest
from lace.cache import sc
from lace.serialization import wrl, obj

class TestWRL(unittest.TestCase):
    def setUp(self):
        self.test_wrl_url = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_wrl.wrl"
        self.test_wrl_path = sc(self.test_wrl_url)
        self.test_obj_url = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_box.obj"
        self.test_wrl_converted_path = sc("s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_wrl_converted.obj")

    def test_loads_from_open_file_using_serializer(self):
        with open(self.test_wrl_path) as f:
            m = wrl.load(f)
        with open(self.test_wrl_converted_path) as f:
            m_truth = obj.load(f)
        self.assertTrue((m.v == m_truth.v).all())
        self.assertTrue((m.f == m_truth.f).all())

    def test_loads_unsupported_format_raise_exception(self):
        with self.assertRaises(wrl.ParseError):
            with open(sc(self.test_obj_url)) as f:
                wrl.load(f)
