import unittest
import os
from textwrap import dedent
import numpy as np
from bltest import attr
from baiji import s3
from bltest import skip_if_unavailable, skip_on_import_error
from bltest.extra_asserts import ExtraAssertionsMixin
import mock
from lace.mesh import Mesh
from lace.serialization import obj
from lace.cache import sc
from lace.cache import vc
from .testing.scratch_dir import ScratchDirMixin

@attr('missing_assets')
class TestOBJBase(ExtraAssertionsMixin, unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp('bodylabs-test')
        self.truth = {
            'box_v': np.array([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]]).T,
            'box_f': np.array([[0, 1, 2], [3, 2, 1], [0, 2, 4], [6, 4, 2], [0, 4, 1], [5, 1, 4], [7, 5, 6], [4, 6, 5], [7, 6, 3], [2, 3, 6], [7, 3, 5], [1, 5, 3]]),
            'box_segm': {'a':np.array(range(6), dtype=np.uint32), 'b':np.array([6, 10, 11], dtype=np.uint32), 'c':np.array([7, 8, 9], dtype=np.uint32)},
            'box_segm_overlapping': {'a':np.array(range(6), dtype=np.uint32), 'b':np.array([6, 10, 11], dtype=np.uint32), 'c':np.array([7, 8, 9], dtype=np.uint32), 'd':np.array([1, 2, 8], dtype=np.uint32)},
            'landm': {'pospospos' : 0, 'negnegneg' : 7},
            'landm_xyz': {'pospospos' : np.array([0.5, 0.5, 0.5]), 'negnegneg' : np.array([-0.5, -0.5, -0.5])},
        }
        self.test_obj_url = vc.uri('/unittest/serialization/obj/test_box_simple.obj')
        self.test_obj_path = vc('/unittest/serialization/obj/test_box_simple.obj')
        self.test_obj_with_vertex_colors_url = vc.uri('/unittest/serialization/obj/test_box_simple_with_vertex_colors.obj')
        self.test_obj_with_landmarks_url = vc.uri('/unittest/serialization/obj/test_box.obj')
        self.test_obj_with_landmarks_path = vc('/unittest/serialization/obj/test_box.obj')
        self.test_pp_path = vc('/unittest/serialization/obj/test_box.pp')
        self.test_obj_with_overlapping_groups_path = vc('/unittest/serialization/obj/test_box_with_overlapping_groups.obj')

        self.obj_with_texure = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/textured_mean_scape_female.obj"
        self.obj_with_texure_mtl = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/textured_mean_scape_female.mtl"
        self.obj_with_texure_tex = "s3://bodylabs-korper-assets/is/ps/shared/data/body/korper_testdata/textured_mean_scape_female.png"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

class TestOBJBasicLoading(TestOBJBase):

    def test_loads_from_local_path_using_constructor(self):
        m = Mesh(filename=self.test_obj_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm'])
        self.assertEqual(m.materials_filepath, None)

    def test_loads_from_local_path_using_serializer(self):
        m = obj.load(self.test_obj_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm'])

    def test_loads_from_remote_path_using_serializer(self):
        skip_if_unavailable('s3')
        m = obj.load(self.test_obj_url)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm'])

    def test_loads_from_open_file_using_serializer(self):
        with open(self.test_obj_path) as f:
            m = obj.load(f)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm'])

    def test_loading_brings_in_normals_and_uvs(self):
        # This file is known to have vt, vn, and faces of the form 1/2/3
        texture_template = 's3://bodylabs-korper-assets/is/ps/shared/data/body/template/texture_coordinates/textured_template_low_v2.obj'
        mesh_with_texture = obj.load(sc(texture_template))
        self.assertIsNotNone(mesh_with_texture.vt)
        self.assertIsNotNone(mesh_with_texture.ft)
        self.assertEqual(mesh_with_texture.vt.shape[1], 2)
        self.assertEqual(mesh_with_texture.vt.shape[0], np.max(mesh_with_texture.ft)+1)
        self.assertIsNotNone(mesh_with_texture.vn)
        self.assertIsNotNone(mesh_with_texture.fn)
        self.assertEqual(mesh_with_texture.vn.shape[1], 3)
        self.assertEqual(mesh_with_texture.vn.shape[0], np.max(mesh_with_texture.fn)+1)

    def test_loading_vertex_colors(self):
        # Mesh without vertex colors should not have vertex colors
        mesh_without_vertex_colors = obj.load(sc(self.test_obj_url))
        self.assertIsNone(mesh_without_vertex_colors.vc)

        # Mesh with vertex colors should have vertex colors
        mesh_with_vertex_colors = obj.load(sc(self.test_obj_with_vertex_colors_url))
        self.assertIsNotNone(mesh_with_vertex_colors.vc)

        # Check sizes
        vc_length, vc_size = mesh_with_vertex_colors.vc.shape
        v_length, _ = mesh_with_vertex_colors.v.shape
        self.assertEqual(vc_length, v_length)
        self.assertEqual(vc_size, 3)

        # Vertices should be the same
        self.assertTrue((mesh_without_vertex_colors.v == mesh_with_vertex_colors.v).all())


class TestOBJWithLandmarks(TestOBJBase):

    def test_loads_from_local_path_using_constructor_with_landmarks(self):
        skip_on_import_error('lace-search')
        m = Mesh(filename=self.test_obj_with_landmarks_path, ppfilename=self.test_pp_path)
        self.assertTrue((m.v == self.truth['box_v']).all())
        self.assertTrue((m.f == self.truth['box_f']).all())
        self.assertEqual(m.landm, self.truth['landm'])
        self.assertDictOfArraysEqual(m.landm_xyz, self.truth['landm_xyz'])
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm'])

class TestOBJBasicWriting(TestOBJBase):

    def test_writing_obj_locally_using_mesh_write_obj(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ascii_obj_locally_using_mesh_write_ply.obj")
        m = Mesh(filename=self.test_obj_path)
        m.write_obj(local_file)
        self.assertFilesEqual(local_file, self.test_obj_path)

    def test_writing_obj_locally_using_serializer(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ascii_obj_locally_using_serializer.obj")
        m = Mesh(filename=self.test_obj_path)
        obj.dump(m, local_file)
        self.assertFilesEqual(local_file, self.test_obj_path)

class TestOBJWithMaterials(ScratchDirMixin, TestOBJBase):

    def test_writing_obj_with_mtl(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_obj_with_mtl.obj")
        m = obj.load(sc(self.obj_with_texure))
        obj.dump(m, local_file)
        self.assertTrue(s3.exists(os.path.splitext(local_file)[0] + '.mtl'))
        self.assertTrue(s3.exists(os.path.splitext(local_file)[0] + '.png'))

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    def test_reading_obj_with_mtl_from_local_file(self, mock_s3_open):
        local_obj_with_texure = os.path.join(self.tmp_dir, os.path.basename(self.obj_with_texure))
        local_obj_with_texure_mtl = os.path.join(self.tmp_dir, os.path.basename(self.obj_with_texure_mtl))
        local_obj_with_texure_tex = os.path.join(self.tmp_dir, os.path.basename(self.obj_with_texure_tex))
        s3.cp(sc(self.obj_with_texure), local_obj_with_texure)
        s3.cp(sc(self.obj_with_texure_mtl), local_obj_with_texure_mtl)
        s3.cp(sc(self.obj_with_texure_tex), local_obj_with_texure_tex)

        m = obj.load(local_obj_with_texure)

        mock_s3_open.assert_has_calls([
            mock.call(local_obj_with_texure, 'rb'),
            mock.call(local_obj_with_texure_mtl, 'r'),
        ])
        self.assertEqual(m.materials_filepath, local_obj_with_texure_mtl)
        self.assertEqual(m.texture_filepath, local_obj_with_texure_tex)

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    # @mock.patch('baiji.pod.asset_cache.AssetCache.__call__', side_effect=sc.__call__)
    def test_reading_obj_with_mtl_from_sc_file(self, mock_sc, mock_s3_open):
        from baiji.pod.asset_cache import CacheFile

        sc_obj_with_texure = self.obj_with_texure.replace("s3://bodylabs-korper-assets", '')
        sc_obj_with_texure_mtl = self.obj_with_texure_mtl.replace("s3://bodylabs-korper-assets", '')
        sc_obj_with_texure_tex = self.obj_with_texure_tex.replace("s3://bodylabs-korper-assets", '')
        bucket = "bodylabs-korper-assets"

        m = obj.load(sc(sc_obj_with_texure, bucket=bucket))

        mock_sc.assert_has_calls([
            mock.call(sc_obj_with_texure, bucket=bucket), # the one above
            mock.call(CacheFile(sc, sc_obj_with_texure_mtl, bucket=bucket).local), # in obj.load
            mock.call(CacheFile(sc, sc_obj_with_texure_tex, bucket=bucket).local), # in obj.load
        ])
        mock_s3_open.assert_has_calls([
            mock.call(sc(sc_obj_with_texure, bucket=bucket), 'rb'),
            mock.call(sc(sc_obj_with_texure_mtl, bucket=bucket), 'r'),
        ])
        self.assertEqual(m.materials_filepath, sc(sc_obj_with_texure_mtl, bucket=bucket))
        self.assertEqual(m.texture_filepath, sc(sc_obj_with_texure_tex, bucket=bucket))

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    def test_reading_obj_with_mtl_from_s3_url(self, mock_s3_open):
        skip_if_unavailable('s3')
        m = obj.load(self.obj_with_texure)

        mock_s3_open.assert_has_calls([
            mock.call(self.obj_with_texure, 'rb'),
            mock.call(self.obj_with_texure_mtl, 'r'),
        ])
        self.assertEqual(m.materials_filepath, self.obj_with_texure_mtl)
        self.assertEqual(m.texture_filepath, self.obj_with_texure_tex)
        self.assertIsNotNone(m.texture_image)

    def test_changing_texture_filepath(self):
        m = obj.load(self.obj_with_texure)
        self.assertEqual(m.texture_filepath, self.obj_with_texure_tex)
        self.assertIsNotNone(m.texture_image)
        m.texture_filepath = None
        self.assertIsNone(m.texture_image)

    def create_texture_test_files(self, include_Ka=False, include_Kd=False):
        obj_path = os.path.join(self.scratch_dir, 'texture_test.obj')
        with open(obj_path, 'w') as f:
            f.write('mtllib {}\n'.format('texture_test.mtl'))
        mtl_path = os.path.join(self.scratch_dir, 'texture_test.mtl')
        with open(mtl_path, 'w') as f:
            if include_Ka:
                f.write('map_Ka {}\n'.format('ambient_tex.png'))
            if include_Kd:
                f.write('map_Kd {}\n'.format('diffuse_tex.png'))
        return obj_path

    def test_texture_reads_Ka(self):
        obj_path = self.create_texture_test_files(include_Ka=True)
        m = obj.load(obj_path)
        self.assertEqual(m.texture_filepath, os.path.join(self.scratch_dir, 'ambient_tex.png'))

    def test_texture_reads_Kd(self):
        obj_path = self.create_texture_test_files(include_Kd=True)
        m = obj.load(obj_path)
        self.assertEqual(m.texture_filepath, os.path.join(self.scratch_dir, 'diffuse_tex.png'))

    def test_texture_reads_Ka_if_both_Ka_and_Kd_are_present(self):
        obj_path = self.create_texture_test_files(include_Ka=True, include_Kd=True)
        m = obj.load(obj_path)
        self.assertEqual(m.texture_filepath, os.path.join(self.scratch_dir, 'ambient_tex.png'))

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    def test_reading_obj_with_mtl_from_absolute_path(self, mock_s3_open):
        # This is generally a very bad idea; it makes it hard to move an obj around
        skip_if_unavailable('s3')
        obj_path = os.path.join(self.scratch_dir, 'abs_path_to_mtl.obj')
        mlt_path = os.path.join(self.scratch_dir, 'abs_path_to_mtl.mlt')
        tex_path = os.path.abspath(sc(self.obj_with_texure_tex))

        with open(obj_path, 'w') as f:
            f.write('mtllib {}\n'.format(mlt_path))
        with open(mlt_path, 'w') as f:
            f.write('map_Ka {}\n'.format(tex_path))

        m = obj.load(obj_path)

        mock_s3_open.assert_has_calls([
            mock.call(obj_path, 'rb'),
            mock.call(mlt_path, 'r'),
        ])
        self.assertEqual(m.materials_filepath, mlt_path)
        self.assertEqual(m.texture_filepath, tex_path)

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    def test_reading_obj_with_mtl_from_missing_absolute_path(self, mock_s3_open):
        # If an absolute path is given and the file is missing, try looking in the same directory;
        # this lets you find the most common intention when an abs path is used.
        skip_if_unavailable('s3')
        obj_path = os.path.join(self.scratch_dir, 'abs_path_to_missing_mtl.obj')
        real_mlt_path = os.path.join(self.scratch_dir, 'abs_path_to_missing_mtl.mlt')
        arbitrary_mlt_path = os.path.join(self.scratch_dir, 'some_other_absolute_path', 'abs_path_to_missing_mtl.mlt')
        tex_path = os.path.abspath(sc(self.obj_with_texure_tex))

        with open(obj_path, 'w') as f:
            f.write('mtllib {}\n'.format(arbitrary_mlt_path))
        with open(real_mlt_path, 'w') as f:
            f.write('map_Ka {}\n'.format(tex_path))

        m = obj.load(obj_path)

        mock_s3_open.assert_has_calls([
            mock.call(obj_path, 'rb'),
            mock.call(real_mlt_path, 'r'),
        ])
        self.assertEqual(m.materials_filepath, real_mlt_path)
        self.assertEqual(m.texture_filepath, tex_path)

    @mock.patch('baiji.s3.open', side_effect=s3.open)
    def test_reading_obj_with_mtl_from_missing_windows_absolute_path(self, mock_s3_open):
        # In this case, we're given a windows absolute path, which it totally wrong, but if there happens
        # to be a mtl file of the right name in the same dir as the obj, go for it.
        # This is a signiicant case, because 3dMD outputs mtllib this way.
        skip_if_unavailable('s3')
        obj_path = os.path.join(self.scratch_dir, 'abs_path_to_missing_windows_mtl.obj')
        real_mlt_path = os.path.join(self.scratch_dir, 'abs_path_to_missing_windows_mtl.mlt')
        arbitrary_mlt_path = 'C:/Users/ARGH/Documents/I-Did_some_scans/Subject_47/abs_path_to_missing_windows_mtl.mlt'
        tex_path = os.path.abspath(sc(self.obj_with_texure_tex))

        with open(obj_path, 'w') as f:
            f.write('mtllib {}\n'.format(arbitrary_mlt_path))
        with open(real_mlt_path, 'w') as f:
            f.write('map_Ka {}\n'.format(tex_path))

        m = obj.load(obj_path)

        mock_s3_open.assert_has_calls([
            mock.call(obj_path, 'rb'),
            mock.call(real_mlt_path, 'r'),
        ])
        self.assertEqual(m.materials_filepath, real_mlt_path)
        self.assertEqual(m.texture_filepath, tex_path)

class TestOBJWithComments(TestOBJBase):

    def test_writing_obj_with_no_comments_does_not_write_comments(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ply_with_no_comments_does_not_write_comments.ply")
        m = obj.load(self.test_obj_path)
        obj.dump(m, local_file)
        with open(local_file) as f:
            self.assertNotRegexpMatches(f.read(), '#')

    def test_writing_obj_with_comments_does_write_comments(self):
        local_file = os.path.join(self.tmp_dir, "test_writing_ply_with_comments_does_write_comments.ply")
        m = obj.load(self.test_obj_path)
        obj.dump(m, local_file, comments=['foo bar', 'this is a comment'])
        with open(local_file) as f:
            contents = f.read()
            self.assertRegexpMatches(contents, '# foo bar\n# this is a comment\n')
            self.assertNotRegexpMatches(contents, '# Copyright')

class TestOBJSpecialCases(TestOBJBase):

    def test_writing_segmented_mesh_preserves_face_order(self):
        m = obj.load(self.test_obj_path)
        self.assertTrue((m.f == self.truth['box_f']).all())

        local_file = os.path.join(self.tmp_dir, 'test_writing_segmented_mesh_preserves_face_order.obj')
        obj.dump(m, local_file)
        m_reloaded = obj.load(local_file)

        self.assertTrue((m_reloaded.f == self.truth['box_f']).all())

    def test_read_overlapping_groups(self):
        m = obj.load(self.test_obj_with_overlapping_groups_path)
        self.assertDictOfArraysEqual(m.segm, self.truth['box_segm_overlapping'])

    def test_write_overlapping_groups(self):
        m = obj.load(self.test_obj_with_overlapping_groups_path)

        local_file = os.path.join(self.tmp_dir, 'test_write_overlapping_groups.obj')
        obj.dump(m, local_file)

        self.assertFilesEqual(local_file, self.test_obj_with_overlapping_groups_path)

    def test_writing_mesh_with_overlapping_segments_preserves_face_order(self):
        '''
        Covered by test above, but covered here in a less fragile way, for good measure.

        '''
        m = obj.load(self.test_obj_with_overlapping_groups_path)
        self.assertTrue((m.f == self.truth['box_f']).all())

        local_file = os.path.join(self.tmp_dir, 'test_writing_mesh_with_overlapping_segments_preserves_face_order.obj')
        obj.dump(m, local_file)
        m_reloaded = obj.load(local_file)

        self.assertTrue((m_reloaded.f == self.truth['box_f']).all())

    def test_writing_empty_mesh(self):
        m = Mesh()

        local_file = os.path.join(self.tmp_dir, 'test_writing_empty_mesh.obj')
        obj.dump(m, local_file)

        self.assertEqual(os.stat(local_file).st_size, 0)

class TestOBJDangerousInputs(TestOBJBase):
    '''
    Here we test different malformations that could be dangerous to feed to the obj parser.
    We're not testing what gets loaded; we'te testing that appropriate exceptions are thrown
    or we pass without failure
    '''

    def write_then_load(self, contents):
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            f.write(contents)
            f.flush()
            m = obj.load(f.name)
            return m

    def test_empty_file(self):
        m = self.write_then_load('')
        self.assertEqual(m.v, None)
        self.assertEqual(m.f, None)

    def test_junk(self):
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load(dedent('''
            dog
            cat
            rabbit
            fox
            ''').lstrip())

    def test_blanks_and_comments(self):
        self.write_then_load("  \n") # spaces on a blank line
        self.write_then_load("\t\n") # tabs on a blank line
        self.write_then_load("#foo\n") # comments
        self.write_then_load("\n\n") # consecutive newlines
        self.write_then_load("\n\r") # windows file endings
        self.write_then_load("\r\n") # I can never remember which order they go in

    def test_tags_without_delimiters_before_data_is_malformed(self):
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('v1 2 3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vn1 2 3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vt1 2')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('f1 2 3')

    def test_tags_with_extra_characters_are_malformed(self):
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vx 1 2 3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vnx 1 2 3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vtx 1 2')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('fx 1 2 3')

    def test_verticies_must_be_numbers(self):
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('v x y z')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vt u v')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('vn x y z')

    def test_faces_must_be_well_formed(self):
        self.write_then_load('f 1 2 3')
        self.write_then_load('f 1/1 2/2 3/3')
        self.write_then_load('f 1//1 2//2 3//3')
        self.write_then_load('f 1/1/1 2/2/2 3/3/3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('f x y z')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('f /1 /2 /3')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('f 1/1/ 2/2/ 3/3/')
        with self.assertRaises(obj.LoadObjError):
            self.write_then_load('f 1/ 2/ 3/')

    def test_allowed_tags(self):
        self.write_then_load("mtllib foobar")
        self.write_then_load("g foobar")
        self.write_then_load("#landmark foobar")
        self.write_then_load("usemtl foobar")
        self.write_then_load("vp 1 2 3")
        self.write_then_load("o foobar")
        self.write_then_load("s 1")
        self.write_then_load("s off")
        self.write_then_load("s") # really 3dMD? what does this even mean?


if __name__ == '__main__': # pragma: no cover
    unittest.main()
