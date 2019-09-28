class ScratchDirMixin(object):
    """
    Abstract Test case which defines and creates a `scratch_dir` on startup,
    and deletes it when finished.
    """
    def setUp(self):
        import tempfile
        super(ScratchDirMixin, self).setUp()
        self.scratch_dir = tempfile.mkdtemp('scratch-dir')

    def tearDown(self):
        import shutil
        super(ScratchDirMixin, self).tearDown()
        shutil.rmtree(self.scratch_dir)

    def assertExists(self, path):
        import os
        self.assertTrue(
            os.path.exists(path),
            msg="Expected file '{}' does not exist.".format(path))

    def get_tmp_path(self, *paths):
        import os
        return os.path.join(self.scratch_dir, *paths)
