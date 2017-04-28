import unittest
import os
import numpy as np
from bltest import skip_if_unavailable, skip_on_import_error
from bltest.extra_asserts import ExtraAssertionsMixin
from lace.mesh import Mesh
from lace.serialization import ply

class TestPLYBase(ExtraAssertionsMixin, unittest.TestCase):
    def setUp(self):
        import tempfile
        from lace.cache import sc
        self.tmp_dir = tempfile.mkdtemp('bodylabs-test')
        self.truth = {
            'box_v': np.array([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]]).T,
            'box_f': np.array([[0, 1, 2], [3, 2, 1], [0, 2, 4], [6, 4, 2], [0, 4, 1], [5, 1, 4], [7, 5, 6], [4, 6, 5], [7, 6, 3], [2, 3, 6], [7, 3, 5], [1, 5, 3]]),
            'box_segm': {'a':np.array(range(6), dtype=np.uint32), 'b':np.array([6, 10, 11], dtype=np.uint32), 'c':np.array([7, 8, 9], dtype=np.uint32)},
            'landm': {'pospospos' : 0, 'negnegneg' : 7},
            'landm_xyz': {'pospospos' : np.array([0.5, 0.5, 0.5]), 'negnegneg' : np.array([-0.5, -0.5, -0.5])},
        }
        self.test_ply_url = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_box.ply"
        self.test_ply_path = sc(self.test_ply_url)
        self.test_bin_ply_path = sc("s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_box_le.ply")
        self.test_pp_path = sc("s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/test_box.pp")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

class TestPLY(TestPLYBase):

    def test_loads_from_local_path_using_explicit_constructor(self):
        m = Mesh(filename=self.test_ply_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())

    def test_loads_from_local_path_using_implicit_constructor(self):
        m = Mesh(self.test_ply_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())

    def test_loads_from_local_path_using_serializer(self):
        m = ply.load(self.test_ply_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())

    def test_loads_from_remote_path_using_serializer(self):
        skip_if_unavailable('s3')
        m = ply.load(self.test_ply_url)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())

    def test_loads_from_open_file_using_serializer(self):
        with open(self.test_ply_path) as f:
            m = ply.load(f)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())

    def test_loads_from_local_path_using_constructor_with_landmarks(self):
        skip_on_import_error('lace-search')
        m = Mesh(filename=self.test_ply_path, ppfilename=self.test_pp_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertEqual(m.landm, self.truth['landm'])

    def test_writing_ascii_ply_locally_using_mesh_write_ply(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ascii_ply_locally_using_mesh_write_ply.ply")
        m = Mesh(filename=self.test_ply_path)
        m.write_ply(local_file, ascii=True)
        self.assertFilesEqual(local_file, self.test_ply_path)

    def test_writing_binary_ply_locally_using_mesh_write_ply(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ascii_ply.ply")
        m = Mesh(filename=self.test_ply_path)
        m.write_ply(local_file)
        self.assertFilesEqual(local_file, self.test_bin_ply_path)

    def test_writing_ascii_ply_locally_using_serializer(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ascii_ply_locally_using_serializer.ply")
        m = Mesh(filename=self.test_ply_path)
        ply.dump(m, local_file, ascii=True)
        self.assertFilesEqual(local_file, self.test_ply_path)

    def test_writing_binary_ply_locally_using_serializer(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_binary_ply_locally_using_serializer.ply")
        m = Mesh(filename=self.test_ply_path)
        ply.dump(m, local_file)
        self.assertFilesEqual(local_file, self.test_bin_ply_path)

    def test_writing_ply_with_no_comments_does_not_write_comments(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ply_with_no_comments_does_not_write_comments.ply")
        m = ply.load(self.test_ply_path)
        ply.dump(m, local_file, ascii=True)
        with open(local_file) as f:
            self.assertNotRegexpMatches(f.read(), '\ncomment')

    def test_writing_ply_with_comments_does_write_comments(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ply_with_comments_does_write_comments.ply")
        m = ply.load(self.test_ply_path)
        ply.dump(m, local_file, ascii=True, comments=['foo bar', 'this is a comment'])
        with open(local_file) as f:
            contents = f.read()
            self.assertRegexpMatches(contents, '\ncomment foo bar\ncomment this is a comment\n')
            self.assertNotRegexpMatches(contents, '\ncomment Copyright')


class TestPLYDangerousInputs(TestPLYBase):
    '''
    Here we test different malformations that could be dangerous to feed to the obj parser.
    We're not testing what gets loaded; we'te testing that appropriate exceptions are thrown
    or we pass without failure
    '''
    def test_hey_look_its_a_buffer_overflow_in_rply(self):
        '''
        Without the rply fix, python will crash executing this (on osx, it
        crashes with "Abort trap: 6"). This test is passing when it doesn't
        crash :)

        The bug is in rply's error handling, which trustingly writes the name
        of an element into a fixed length buffer when there's a problem reading
        the element or its properties.
        '''
        bad_file = os.path.join(self.tmp_dir, 'danger_will_robinson.ply')
        with open(bad_file, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vert' + (1024*'e') + 'x 8\n')
            f.write('end_header\n')
        with self.assertRaises(ply.PLYError):
            ply.load(bad_file)

if __name__ == '__main__': # pragma: no cover
    unittest.main()
