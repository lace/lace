class MeshMixin(object):
    def show(self, mv=None, meshes=None, lines=None, with_measurements=False, titlebar='Mesh Viewer'):
        import numpy as np
        from lace.meshviewer import MeshViewer
        from blmath.numerics.matlab import row
        if meshes is None:
            meshes = []
        if lines is None:
            lines = []
        if mv is None:
            mv = MeshViewer(keepalive=True, titlebar=titlebar)
        else:
            mv.titlebar = titlebar
        if self.landm is not None:
            from lace.sphere import Sphere
            sphere = Sphere()
            scalefactor = 1e-2 * np.max(np.max(self.v) - np.min(self.v)) / np.max(np.max(sphere.v) - np.min(sphere.v))
            sphere.scale(scalefactor)
            spheres = [self.__class__(vc='SteelBlue', f=sphere.f, v=sphere.v + row(np.array(self.landm_xyz[k]))) for k in self.landm.keys()]
            mv.set_dynamic_meshes([self] + spheres + meshes, blocking=True)
        else:
            mv.set_dynamic_meshes([self] + meshes, blocking=True)
        if with_measurements:
            lines.extend([line for m in self.measurements for line in m.lines])
        mv.set_dynamic_lines(lines)
        return mv
