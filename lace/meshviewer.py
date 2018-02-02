#!/usr/bin/env python
# pylint: disable=attribute-defined-outside-init
# pylint: disable=invalid-unary-operand-type, too-many-lines
# encoding: utf-8

import sys
import os
import os.path
import time
import copy
import subprocess
import re
import platform
from multiprocessing import freeze_support
import numpy as np
import OpenGL.GL as gl
from OpenGL.GL import shaders
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import OpenGL.arrays.vbo
from blmath.numerics.matlab import row
from blmath.optimization.objectives.normals import TriNormals
import zmq
from lace import arcball
from lace.mesh import Mesh

__all__ = ['MeshViewer', 'MeshViewers', 'test_for_opengl']

"""
MeshViewer.py

Created by Matthew Loper on 2012-05-11.
Copyright (c) 2012 MPI. All rights reserved.
"""


def _popen_exec_python(command, args=[], stdin=None, stdout=None, stderr=None):
    return subprocess.Popen(
        [sys.executable, command] + args,
        stdin=stdin, stdout=stdout, stderr=stderr
    )


def _run_self(args, stdin=None, stdout=subprocess.PIPE, stderr=None):
    if platform.system() == 'Windows':
        try:
            from py_windows_exe import running_as_windows_exe, get_main_dir
            if running_as_windows_exe():
                env = os.environ.copy()
                env["PATH"] = "%s;%s" % (get_main_dir(), env["PATH"])
                return subprocess.Popen(
                    [os.path.join(get_main_dir(), 'pymeshviewer.exe')] + args,
                    stdin=stdin, stdout=stdout, stderr=stderr, env=env
                )
        except Exception:  # pylint: disable=broad-except
            pass
    return _popen_exec_python(
        os.path.abspath(__file__), args, stdin=stdin, stdout=stdout, stderr=stderr
    )


def test_for_opengl():
    if test_for_opengl.result is None:
        p = _run_self(["TEST_FOR_OPENGL"], stderr=open(os.devnull, 'wb'))
        line = p.stdout.readline()
        test_for_opengl.result = 'success' in line
    return test_for_opengl.result


test_for_opengl.result = None


class Dummy(object):
    def __getattr__(self, name):
        return Dummy()

    def __call__(self, *args, **kwargs):
        return Dummy()

    def __getitem__(self, key):
        return Dummy()

    def __setitem__(self, key, value):
        pass


def MeshViewer(
        titlebar='Mesh Viewer', static_meshes=None, static_lines=None, uid=None,
        autorecenter=True, keepalive=False, window_width=1280, window_height=960,
        snapshot_camera=None
):
    """Allows visual inspection of geometric primitives.

    Write-only Attributes:
        titlebar: string printed in the window titlebar
        dynamic_meshes: list of Mesh objects to be displayed
        static_meshes: list of Mesh objects to be displayed
        dynamic_lines: list of Lines objects to be displayed
        static_lines: list of Lines objects to be displayed

    Note: static_meshes is meant for Meshes that are
    updated infrequently, and dynamic_meshes is for Meshes
    that are updated frequently (same for dynamic_lines vs
    static_lines). They may be treated differently for
    performance reasons.
    """
    if not test_for_opengl():
        return Dummy()
    mv = MeshViewerLocal(
        shape=(1, 1), uid=uid, titlebar=titlebar, keepalive=keepalive,
        window_width=window_width, window_height=window_height
    )
    result = mv.get_subwindows()[0][0]
    result.snapshot_camera = snapshot_camera
    if static_meshes:
        result.static_meshes = static_meshes
    if static_lines:
        result.static_lines = static_lines
    result.autorecenter = autorecenter
    return result


def MeshViewers(
        shape=(1, 1), titlebar="Mesh Viewers", keepalive=False,
        window_width=1280, window_height=960
):
    """Allows subplot-style inspection of primitives in multiple subwindows.

    Args:
        shape: a tuple indicating the number of vertical and horizontal windows requested

    Returns: a list of lists of MeshViewer objects: one per window requested.
    """
    if not test_for_opengl():
        return Dummy()
    mv = MeshViewerLocal(
        shape=shape, titlebar=titlebar, uid=None, keepalive=keepalive,
        window_width=window_width, window_height=window_height
    )
    return mv.get_subwindows()


class MeshSubwindow(object):
    def __init__(self, parent_window, which_window):
        self.parent_window = parent_window
        self.which_window = which_window

    def set_dynamic_meshes(self, list_of_meshes, blocking=False):
        self.parent_window.set_dynamic_meshes(list_of_meshes, blocking, self.which_window)

    def set_static_meshes(self, list_of_meshes, blocking=False):
        self.parent_window.set_static_meshes(list_of_meshes, blocking, self.which_window)

    # list_of_model_names_and_parameters should be of form
    # [{'name': scape_model_name, 'parameters': scape_model_parameters}]
    # here scape_model_name is the filepath of the scape model.
    def set_dynamic_models(self, list_of_model_names_and_parameters, blocking=False):
        self.parent_window.set_dynamic_models(
            list_of_model_names_and_parameters, blocking, self.which_window
        )

    def set_dynamic_lines(self, list_of_lines, blocking=False):
        self.parent_window.set_dynamic_lines(list_of_lines, blocking, self.which_window)

    def set_static_lines(self, list_of_lines, blocking=False):
        self.parent_window.set_static_lines(
            list_of_lines, blocking=blocking, which_window=self.which_window
        )

    def set_titlebar(self, titlebar, blocking=False):
        self.parent_window.set_titlebar(titlebar, blocking, which_window=self.which_window)

    def set_autorecenter(self, autorecenter, blocking=False):
        self.parent_window.set_autorecenter(
            autorecenter, blocking=blocking, which_window=self.which_window
        )

    def set_background_color(self, background_color, blocking=False):
        self.parent_window.set_background_color(
            background_color, blocking=blocking, which_window=self.which_window
        )

    def save_snapshot(self, path, blocking=False):
        self.parent_window.save_snapshot(path, blocking=blocking, which_window=self.which_window)

    def get_event(self):
        return self.parent_window.get_event()

    def get_keypress(self):
        return self.parent_window.get_keypress()['key']

    def get_mouseclick(self):
        return self.parent_window.get_mouseclick()

    def close(self):
        self.parent_window.p.terminate()

    background_color = property(
        fset=set_background_color,
        doc="Background color, as 3-element numpy array where 0 <= color <= 1.0."
    )
    dynamic_meshes = property(
        fset=set_dynamic_meshes, doc="List of meshes for dynamic display."
    )
    static_meshes = property(
        fset=set_static_meshes, doc="List of meshes for static display."
    )
    dynamic_models = property(
        fset=set_dynamic_models,
        doc="List of model names and parameters for dynamic display."
    )
    dynamic_lines = property(
        fset=set_dynamic_lines, doc="List of Lines for dynamic display."
    )
    static_lines = property(
        fset=set_static_lines, doc="List of Lines for static display."
    )
    titlebar = property(fset=set_titlebar, doc="Titlebar string.")


class MeshViewerLocal(object):
    """Allows visual inspection of geometric primitives.

    Write-only Attributes:
        titlebar: string printed in the window titlebar
        dynamic_meshes: list of Mesh objects to be displayed
        static_meshes: list of Mesh objects to be displayed
        dynamic_lines: list of Lines objects to be displayed
        static_lines: list of Lines objects to be displayed

    Note: static_meshes is meant for Meshes that are
    updated infrequently, and dynamic_meshes is for Meshes
    that are updated frequently (same for dynamic_lines vs
    static_lines). They may be treated differently for
    performance reasons.
    """
    managed = {}

    def __new__(cls, titlebar, uid, shape, keepalive, window_width, window_height):
        import traceback
        assert uid is None or isinstance(uid, str) or isinstance(uid, unicode)
        if uid == 'stack':
            uid = ''.join(traceback.format_list(traceback.extract_stack()))
        if uid and uid in MeshViewer.managed:
            return MeshViewer.managed[uid]
        result = super(MeshViewerLocal, cls).__new__(cls)
        result.client = zmq.Context.instance().socket(zmq.PUSH)
        result.client.linger = 0
        result.p = _run_self(
            [
                titlebar,
                str(shape[0]),
                str(shape[1]),
                str(window_width),
                str(window_height)
            ]
        )
        line = result.p.stdout.readline()
        current_port = re.match('<PORT>(.*?)</PORT>', line)
        if not current_port:
            raise Exception("MeshViewer remote appears to have failed to launch")
        current_port = int(current_port.group(1))
        result.client.connect('tcp://127.0.0.1:%d' % (current_port))
        if uid:
            MeshViewerLocal.managed[uid] = result
        result.shape = shape
        result.keepalive = keepalive
        return result

    def get_subwindows(self):
        return [
            [
                MeshSubwindow(parent_window=self, which_window=(r, c))
                for c in range(self.shape[1])
            ]
            for r in range(self.shape[0])
        ]

    @staticmethod
    def _sanitize_meshes(list_of_meshes):
        lm = []
        # have to copy the meshes for now, because some contain CPython members,
        # before pushing them on the queue
        for m in list_of_meshes:
            if m.f is not None:
                f = m.f
                if m.fc is not None:
                    lm.append(Mesh(v=m.v, f=f, fc=m.fc))
                elif m.vc is not None:
                    lm.append(Mesh(v=m.v, f=f, vc=m.vc))
                else:
                    lm.append(Mesh(v=m.v, f=f))
            else:
                if m.fc is not None:
                    lm.append(Mesh(v=m.v, fc=m.fc))
                elif m.vc is not None:
                    lm.append(Mesh(v=m.v, vc=m.vc))
                else:
                    lm.append(Mesh(v=m.v))
            if m.vn is not None:
                lm[-1].vn = m.vn
            if m.fn is not None:
                lm[-1].fn = m.fn
            if m.texture_filepath is not None and m.vt is not None and m.ft is not None:
                lm[-1].texture_filepath = m.texture_filepath
                lm[-1].vt = m.vt
                lm[-1].ft = m.ft
            if m.landm_xyz is not None and m.landm is not None:
                from lace.sphere import Sphere
                sphere = Sphere()
                scalefactor = (
                    1e-2 * np.max(np.max(m.v) - np.min(m.v)) / np.max(
                        np.max(sphere.v) - np.min(sphere.v)
                    )
                )
                sphere.scale(scalefactor)
                spheres = [Mesh(vc='SteelBlue', f=sphere.f,
                                v=sphere.v + row(np.array(m.landm_xyz[k])))
                           for k in m.landm.keys()]
                lm.extend(spheres)
        return lm

    def _send_pyobj(self, label, obj, blocking, which_window):
        if blocking:
            context = zmq.Context.instance()
            server = context.socket(zmq.PULL)
            server.linger = 0
            port = server.bind_to_random_port(
                'tcp://127.0.0.1', min_port=49152, max_port=65535, max_tries=100000
            )
            self.client.send_pyobj(
                {
                    'label': label,
                    'obj': obj,
                    'port': port,
                    'which_window': which_window
                }
            )
            task_completion_time = server.recv_pyobj()  # pylint: disable=unused-variable
            server.close()
        else:

            self.client.send_pyobj({'label': label, 'obj': obj, 'which_window': which_window})

    def set_dynamic_meshes(self, list_of_meshes, blocking=False, which_window=(0, 0)):
        self._send_pyobj(
            'dynamic_meshes', self._sanitize_meshes(list_of_meshes),
            blocking, which_window
        )

    def set_static_meshes(self, list_of_meshes, blocking=False, which_window=(0, 0)):
        self._send_pyobj(
            'static_meshes', self._sanitize_meshes(list_of_meshes),
            blocking, which_window
        )

    # list_of_model_names_and_parameters should be of form
    # [{'name': scape_model_name, 'parameters': scape_model_parameters}]
    # here scape_model_name is the filepath of the scape model.
    def set_dynamic_models(
            self, list_of_model_names_and_parameters,
            blocking=False, which_window=(0, 0)
    ):
        self._send_pyobj(
            'dynamic_models', list_of_model_names_and_parameters,
            blocking, which_window
        )

    def set_dynamic_lines(self, list_of_lines, blocking=False, which_window=(0, 0)):
        self._send_pyobj('dynamic_lines', list_of_lines, blocking, which_window)

    def set_static_lines(self, list_of_lines, blocking=False, which_window=(0, 0)):
        self._send_pyobj('static_lines', list_of_lines, blocking, which_window)

    def set_titlebar(self, titlebar, blocking=False, which_window=(0, 0)):
        self._send_pyobj('titlebar', titlebar, blocking, which_window)

    def set_autorecenter(self, autorecenter, blocking=False, which_window=(0, 0)):
        self._send_pyobj('autorecenter', autorecenter, blocking, which_window)

    def set_background_color(self, background_color, blocking=False, which_window=(0, 0)):
        assert isinstance(background_color, np.ndarray)
        assert background_color.size == 3
        self._send_pyobj('background_color', background_color.flatten(), blocking, which_window)

    def get_keypress(self):
        return self.get_ui_event('get_keypress')

    def get_mouseclick(self):
        return self.get_ui_event('get_mouseclick')

    def get_event(self):
        return self.get_ui_event('get_event')

    def get_ui_event(self, event_id):
        context = zmq.Context.instance()
        server = context.socket(zmq.PULL)
        server.linger = 0
        port = server.bind_to_random_port(
            'tcp://127.0.0.1', min_port=49152, max_port=65535, max_tries=100000
        )
        self._send_pyobj(event_id, port, blocking=True, which_window=(0, 0))
        result = server.recv_pyobj()
        server.close()
        return result

    background_color = property(
        fset=set_background_color,
        doc="Background color, as 3-element numpy array where 0 <= color <= 1.0."
    )
    dynamic_meshes = property(
        fset=set_dynamic_meshes, doc="List of meshes for dynamic display."
    )
    static_meshes = property(
        fset=set_static_meshes, doc="List of meshes for static display."
    )
    dynamic_models = property(
        fset=set_dynamic_models,
        doc="List of model names and parameters for dynamic display."
    )
    dynamic_lines = property(
        fset=set_dynamic_lines, doc="List of Lines for dynamic display."
    )
    static_lines = property(
        fset=set_static_lines, doc="List of Lines for static display."
    )
    titlebar = property(fset=set_titlebar, doc="Titlebar string.")

    def save_snapshot(self, path, blocking=False, which_window=(0, 0)):
        self._send_pyobj('save_snapshot', path, blocking, which_window)

    def __del__(self):
        if not self.keepalive:
            self.p.terminate()


class MeshViewerSingle(object):
    '''
    two_sided_lighting: When True, turn on two-sided lighting. I'm deferring the work
      of figuring out how to pass this option e.g. from mesh.show() through the
      MeshViewerRemote and MeshViewerLocal to this object which needs it.

    '''
    def __init__(self, x1_pct, y1_pct, width_pct, height_pct, two_sided_lighting=False):
        assert width_pct <= 1
        assert height_pct <= 1
        self.dynamic_meshes = []
        self.static_meshes = []
        self.dynamic_models = []
        self.dynamic_lines = []
        self.static_lines = []
        self.scape_models = {}
        self.x1_pct = x1_pct
        self.y1_pct = y1_pct
        self.width_pct = width_pct
        self.height_pct = height_pct
        self.autorecenter = True
        self.two_sided_lighting = two_sided_lighting

    def get_dimensions(self):
        d = {}
        d['window_width'] = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        d['window_height'] = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
        d['subwindow_width'] = self.width_pct * d['window_width']
        d['subwindow_height'] = self.height_pct * d['window_height']
        d['subwindow_origin_x'] = self.x1_pct * d['window_width']
        d['subwindow_origin_y'] = self.y1_pct * d['window_height']
        return d

    def on_draw(self, transform, want_camera=False):
        d = self.get_dimensions()
        gl.glViewport(
            int(d['subwindow_origin_x']),
            int(d['subwindow_origin_y']),
            int(d['subwindow_width']),
            int(d['subwindow_height']))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # Note that the near clip plane is 1 (hither) and the far plane is 1000 (yon)
        glu.gluPerspective(
            45.0,
            float(d['subwindow_width']) / float(d['subwindow_height']),
            1,
            100.0
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        if self.two_sided_lighting:
            gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
        else:
            gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_FALSE)
        gl.glTranslatef(0.0, 0.0, -6.0)
        gl.glPushMatrix()
        gl.glMultMatrixf(transform)
        gl.glColor3f(1.0, 0.75, 0.75)
        if self.autorecenter:
            camera = self.draw_primitives_recentered(want_camera=want_camera)
        else:
            if hasattr(self, 'current_center') and hasattr(self, 'current_scalefactor'):
                camera = self.draw_primitives(
                    scalefactor=self.current_scalefactor,
                    center=self.current_center
                )
            else:
                camera = self.draw_primitives(want_camera=want_camera)
        gl.glPopMatrix()
        if want_camera:
            return camera

    def draw_primitives_recentered(self, want_camera=False):
        return self.draw_primitives(recenter=True, want_camera=want_camera)

    @staticmethod
    def set_shaders(m):
        VERTEX_SHADER = shaders.compileShader("""void main() {
                    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                }""", gl.GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader("""void main() {
                    gl_FragColor = vec4( 0, 1, 0, 1 );
                }""", gl.GL_FRAGMENT_SHADER)
        m.shaders = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

    @staticmethod
    def set_texture(m):
        texture_data = np.array(m.texture_image, dtype='int8')
        m.textureID = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, m.textureID)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB, texture_data.shape[1],
            texture_data.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE,
            texture_data.flatten()
        )
        # must be GL_FASTEST, GL_NICEST or GL_DONT_CARE
        gl.glHint(gl.GL_GENERATE_MIPMAP_HINT, gl.GL_NICEST)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    @staticmethod
    def draw_mesh(m):
        # Supply vertices
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        m.vbo['v'].bind()
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, m.vbo['v'])
        m.vbo['v'].unbind()
        # Supply normals
        if 'vn' in m.vbo:
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            m.vbo['vn'].bind()
            gl.glNormalPointer(gl.GL_FLOAT, 0, m.vbo['vn'])
            m.vbo['vn'].unbind()
        else:
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        # Supply colors
        if 'vc' in m.vbo:
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            m.vbo['vc'].bind()
            gl.glColorPointer(3, gl.GL_FLOAT, 0, m.vbo['vc'])
            m.vbo['vc'].unbind()
        else:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        if ('vt' in m.vbo) and hasattr(m, 'textureID'):
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
            gl.glBindTexture(gl.GL_TEXTURE_2D, m.textureID)
            gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            m.vbo['vt'].bind()
            gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, m.vbo['vt'])
            m.vbo['vt'].unbind()
        else:
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        # Draw
        if np.any(m.f):  # i.e. if it is triangulated
            gl.glEnable(gl.GL_LIGHTING)
            gl.glDrawElementsui(gl.GL_TRIANGLES, np.arange(m.f.size, dtype=np.uint32))
        else:  # not triangulated, so disable lighting
            gl.glDisable(gl.GL_LIGHTING)
            gl.glPointSize(2)
            gl.glDrawElementsui(gl.GL_POINTS, np.arange(len(m.v), dtype=np.uint32))

    @staticmethod
    def draw_lines(ls):
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glLineWidth(1.0)
        allpts = ls.v[ls.e.flatten()].astype(np.float32)
        gl.glVertexPointerf(allpts)
        if ls.vc is not None or ls.ec is not None:
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            if ls.vc is not None:
                gl.glColorPointerf(ls.vc[ls.e.flatten()].astype(np.float32))
            else:
                clrs = np.ones((ls.e.shape[0] * 2, 3)) * np.repeat(ls.ec, 2, axis=0)
                gl.glColorPointerf(clrs)
        else:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDrawElementsui(gl.GL_LINES, np.arange(len(allpts), dtype=np.uint32))

    def generate_dynamic_model_meshes(self, remove_head=False):
        try:
            from bodylabs.scape.scapemodel import ScapeModel
        except ImportError:
            raise ImportError(
                "MeshViewer.generate_dynamic_model_meshes requires access to BodyLabs core."
            )
        for dynamic_model in self.dynamic_models:
            scapemodel_fname = dynamic_model['name']
            if scapemodel_fname not in self.scape_models:
                sm = self.scape_models[scapemodel_fname] = ScapeModel(scapemodel_fname)
                if not hasattr(sm.template, 'fbv_cached'):
                    sm.template.fbv_cached = sm.template.faces_by_vertex()

        model_meshes = []
        for dynamic_model in self.dynamic_models:
            model_parameters = dynamic_model['parameters'].copy()
            if 'trans' in model_parameters:
                auto_translate = model_parameters['trans'] == 'auto'
            else:
                auto_translate = False
            if auto_translate:
                del model_parameters['trans']
            model_mesh = self.scape_models[dynamic_model['name']].mesh_for(**model_parameters)
            if auto_translate:
                model_mesh.v = model_mesh.v - np.array([0.0, model_mesh.v[:, 1].min(), 0.0])
            model_mesh.reset_normals(
                self.scape_models[dynamic_model['name']].template.fbv_cached
            )
            model_mesh.set_vertex_colors('pink')
            model_meshes.append(model_mesh)
        if remove_head:
            for i, model_mesh in enumerate(model_meshes):
                model_mesh.remove_faces(
                    self.scape_models[self.dynamic_models[i]['name']].template.segm['head']
                )
        return model_meshes

    def draw_primitives(self, scalefactor=1.0, center=None, recenter=False, want_camera=False):
        # measure the bounding box of all our primitives, so that we can
        # recenter them in our field of view
        if center is None:
            center = [0.0, 0.0, 0.0]
        if np.any(self.dynamic_models):
            all_meshes = (
                self.static_meshes + self.dynamic_meshes + self.generate_dynamic_model_meshes()
            )
        else:
            all_meshes = self.static_meshes + self.dynamic_meshes
        all_lines = self.static_lines + self.dynamic_lines
        if recenter:
            if not (all_meshes or all_lines):
                return
            for m in all_meshes:
                m.v = m.v.reshape((-1, 3))
            all_verts = np.concatenate(
                [
                    m.v[m.f.flatten()] if np.any(m.f) else m.v[:]
                    for m in all_meshes
                ] + [
                    l.v[l.e.flatten()]
                    for l in all_lines
                ],
                axis=0
            )
            maximum = np.max(all_verts, axis=0)
            minimum = np.min(all_verts, axis=0)
            center = (maximum + minimum) / 2.
            scalefactor = (maximum - minimum) / 4.
            scalefactor = np.max(scalefactor)
        else:
            center = np.array(center)
        self.current_center = center
        self.current_scalefactor = scalefactor
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        # uncomment to add a default rotation
        # (useful when automatically snapshoting kinect data)
        # glRotate(220, 0.0, 1.0, 0.0)
        tf = np.identity(4, 'f') / scalefactor
        tf[:3, 3] = -center / scalefactor
        tf[3, 3] = 1
        cur_mtx = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX).T
        gl.glLoadMatrixf(cur_mtx.dot(tf).T)
        if want_camera:
            result = {
                'modelview_matrix': gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX),
                'projection_matrix': gl.glGetDoublev(gl.GL_PROJECTION_MATRIX),
                'viewport': gl.glGetIntegerv(gl.GL_VIEWPORT)
            }
        else:
            result = None
        for m in all_meshes:
            if not hasattr(m, 'vbo'):
                # Precompute vertex vbo
                fidxs = m.f.flatten() if np.any(m.f) else np.arange(len(m.v))
                allpts = m.v[fidxs].astype(np.float32).flatten()
                vbo = OpenGL.arrays.vbo.VBO(allpts)
                m.vbo = {'v': vbo}
                # Precompute normals vbo
                if m.vn is not None:
                    ns = m.vn.astype(np.float32)
                    ns = ns[m.f.flatten(), :]
                    m.vbo['vn'] = OpenGL.arrays.vbo.VBO(ns.flatten())
                elif np.any(m.f):
                    ns = TriNormals(m.v, m.f).r.reshape(-1, 3)
                    ns = np.tile(ns, (1, 3)).reshape(-1, 3).astype(np.float32)
                    m.vbo['vn'] = OpenGL.arrays.vbo.VBO(ns.flatten())
                # Precompute texture vbo
                if np.any(m.ft):
                    ftidxs = m.ft.flatten()
                    data = m.vt[ftidxs].astype(np.float32)[:, 0:2]
                    data[:, 1] = 1.0 - 1.0 * data[:, 1]
                    m.vbo['vt'] = OpenGL.arrays.vbo.VBO(data)
                # Precompute color vbo
                if m.vc is not None:
                    data = m.vc[fidxs].astype(np.float32)
                    m.vbo['vc'] = OpenGL.arrays.vbo.VBO(data)
                elif m.fc is not None:
                    data = np.tile(m.fc, (1, 3)).reshape(-1, 3).astype(np.float32)
                    m.vbo['vc'] = OpenGL.arrays.vbo.VBO(data)
        for e in all_lines:
            self.draw_lines(e)
        for m in all_meshes:
            if m.texture_image is not None and not hasattr(m, 'textureID'):
                self.set_texture(m)
            self.draw_mesh(m)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        return result


class MeshViewerRemote(object):
    def __init__(
            self, titlebar='Mesh Viewer', subwins_vert=1, subwins_horz=1,
            width=100, height=100
    ):
        context = zmq.Context.instance()
        self.server = context.socket(zmq.PULL)
        self.server.linger = 0
        # Find a port to use. The standard set of "private" ports is 49152 through
        # 65535, as seen in...
        # http://en.wikipedia.org/wiki/Port_(computer_networking)
        port = self.server.bind_to_random_port(
            'tcp://127.0.0.1', min_port=49152, max_port=65535, max_tries=100000
        )
        # Print out our port so that our client can connect to us with it.
        # Flush stdout immediately; otherwise our client could wait forever.
        print '<PORT>%d</PORT>\n' % (port, )
        sys.stdout.flush()
        self.arcball = arcball.ArcBallT(width, height)
        self.transform = arcball.Matrix4fT()
        self.lastrot = arcball.Matrix3fT()
        self.thisrot = arcball.Matrix3fT()
        self.isdragging = False
        self.need_redraw = True
        self.mesh_viewers = [
            [
                MeshViewerSingle(
                    float(c) / (subwins_horz),
                    float(r) / (subwins_vert),
                    1. / subwins_horz,
                    1. / subwins_vert
                )
                for c in range(subwins_horz)
            ]
            for r in range(subwins_vert)
        ]
        self.tm_for_fps = 0.
        self.titlebar = titlebar
        self.activate(width, height)

    def snapshot(self, path):
        import cv2
        self.on_draw()
        x = 0
        y = 0
        width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
        data = gl.glReadPixels(x, y, width, height, gl.GL_BGR, gl.GL_UNSIGNED_BYTE)
        data = np.fromstring(data, dtype=np.uint8)
        cv2.imwrite(
            path, np.flipud(data.reshape((height, width, 3)))
        )

    def activate(self, width, height):
        glut.glutInit(['mesh_viewer'])
        glut.glutInitDisplayMode(
            glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_ALPHA | glut.GLUT_DEPTH
        )
        glut.glutInitWindowSize(width, height)
        glut.glutInitWindowPosition(0, 0)
        self.root_window_id = glut.glutCreateWindow(self.titlebar)
        glut.glutDisplayFunc(self.on_draw)
        glut.glutTimerFunc(100, self.checkQueue, 0)
        glut.glutReshapeFunc(self.on_resize_window)
        glut.glutKeyboardFunc(self.on_keypress)
        glut.glutMouseFunc(self.on_click)
        glut.glutMotionFunc(self.on_drag)
        glut.glutDisplayFunc(self.on_draw)
        self.init_opengl()
        glut.glutMainLoop()  # won't return until process is killed

    def on_drag(self, cursor_x, cursor_y):
        """ Mouse cursor is moving
            Glut calls this function (when mouse button is down)
            and pases the mouse cursor postion in window coords as the mouse moves.
        """
        from blmath.geometry.transform.rodrigues import as_rotation_matrix
        if self.isdragging:
            mouse_pt = arcball.Point2fT(cursor_x, cursor_y)
            # Update End Vector And Get Rotation As Quaternion
            ThisQuat = self.arcball.drag(mouse_pt)
            # Convert Quaternion Into Matrix3fT
            self.thisrot = arcball.Matrix3fSetRotationFromQuat4f(ThisQuat)
            # Use correct Linear Algebra matrix multiplication C = A * B
            # Accumulate Last Rotation Into This One
            self.thisrot = arcball.Matrix3fMulMatrix3f(self.lastrot, self.thisrot)
            # make sure it is a rotation
            self.thisrot = as_rotation_matrix(self.thisrot)
            # Set Our Final Transform's Rotation From This One
            self.transform = arcball.Matrix4fSetRotationFromMatrix3f(self.transform, self.thisrot)
            glut.glutPostRedisplay()
        return

    # The function called whenever a key is pressed.
    # Note the use of Python tuples to pass in: (key, x, y)
    def on_keypress(self, *args):
        key = args[0]
        if hasattr(self, 'event_port'):
            self.keypress_port = self.event_port
            del self.event_port
        if hasattr(self, 'keypress_port'):
            client = zmq.Context.instance().socket(zmq.PUSH)
            client.connect('tcp://127.0.0.1:%d' % (self.keypress_port))
            client.send_pyobj({'event_type': 'keyboard', 'key': key})
            del self.keypress_port

    def on_click(self, button, button_state, cursor_x, cursor_y):
        """ Mouse button clicked.
            Glut calls this function when a mouse button is
            clicked or released.
        """
        self.isdragging = False
        if button == glut.GLUT_LEFT_BUTTON and button_state == glut.GLUT_UP:
            # Left button released
            self.lastrot = copy.copy(self.thisrot)  # Set Last Static Rotation To Last Dynamic One
        elif button == glut.GLUT_LEFT_BUTTON and button_state == glut.GLUT_DOWN:
            # Left button clicked down
            self.lastrot = copy.copy(self.thisrot)  # Set Last Static Rotation To Last Dynamic One
            self.isdragging = True  # Prepare For Dragging
            mouse_pt = arcball.Point2fT(cursor_x, cursor_y)
            self.arcball.click(mouse_pt)  # Update Start Vector And Prepare For Dragging
        elif button == glut.GLUT_RIGHT_BUTTON and button_state == glut.GLUT_DOWN:
            # If a mouse click location was requested, return it to caller
            if hasattr(self, 'event_port'):
                self.mouseclick_port = self.event_port
                del self.event_port
            if hasattr(self, 'mouseclick_port'):
                self.send_mouseclick_to_caller(cursor_x, cursor_y)
        elif button == glut.GLUT_MIDDLE_BUTTON and button_state == glut.GLUT_DOWN:
            # If a mouse click location was requested, return it to caller
            if hasattr(self, 'event_port'):
                self.mouseclick_port = self.event_port
                del self.event_port
            if hasattr(self, 'mouseclick_port'):
                self.send_mouseclick_to_caller(cursor_x, cursor_y, button='middle')
        glut.glutPostRedisplay()

    def send_mouseclick_to_caller(self, cursor_x, cursor_y, button='right'):
        client = zmq.Context.instance().socket(zmq.PUSH)
        client.connect('tcp://127.0.0.1:%d' % (self.mouseclick_port))
        cameras = self.on_draw(want_cameras=True)
        window_height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
        depth_value = gl.glReadPixels(
            cursor_x, window_height - cursor_y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT
        )
        pyobj = {
            'event_type': 'mouse_click_%sbutton' % button,
            'u': None, 'v': None,
            'x': None, 'y': None, 'z': None,
            'subwindow_row': None,
            'subwindow_col': None
        }
        for subwin_row, camera_list in enumerate(cameras):
            for subwin_col, camera in enumerate(camera_list):
                # test for out-of-bounds
                if cursor_x < camera['viewport'][0]:
                    continue
                if cursor_x > (camera['viewport'][0] + camera['viewport'][2]):
                    continue
                if window_height - cursor_y < camera['viewport'][1]:
                    continue
                if window_height - cursor_y > (camera['viewport'][1] + camera['viewport'][3]):
                    continue
                xx, yy, zz = glu.gluUnProject(
                    cursor_x, window_height - cursor_y, depth_value,
                    camera['modelview_matrix'],
                    camera['projection_matrix'],
                    camera['viewport']
                )
                pyobj = {
                    'event_type': 'mouse_click_%sbutton' % button,
                    'u': cursor_x - camera['viewport'][0],
                    'v': window_height - cursor_y - camera['viewport'][1],
                    'x': xx, 'y': yy, 'z': zz,
                    'which_subwindow': (subwin_row, subwin_col)
                }
        client.send_pyobj(pyobj)
        del self.mouseclick_port

    def on_draw(self, want_cameras=False):
        self.tm_for_fps = time.time()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        cameras = []
        for mvl in self.mesh_viewers:
            cameras.append([])
            for mv in mvl:
                cameras[-1].append(mv.on_draw(self.transform, want_cameras))
        gl.glFlush()  # Flush The GL Rendering Pipeline
        glut.glutSwapBuffers()
        self.need_redraw = False
        if want_cameras:
            return cameras

    # Reshape The Window When It's Moved Or Resized
    def on_resize_window(self, Width, Height):
        self.arcball.setBounds(Width, Height)  # *NEW* Update mouse bounds for arcball
        return

    def handle_request(self, request):
        label = request['label']
        obj = request['obj']
        w = request['which_window']
        mv = self.mesh_viewers[w[0]][w[1]]
        # Handle each type of request.
        # Some requests require a redraw, and
        # some don't.
        if label == 'dynamic_meshes':
            mv.dynamic_meshes = obj
            self.need_redraw = True
        elif label == 'dynamic_models':
            mv.dynamic_models = obj
            self.need_redraw = True
        elif label == 'static_meshes':
            mv.static_meshes = obj
            self.need_redraw = True
        elif label == 'dynamic_lines':
            mv.dynamic_lines = obj
            self.need_redraw = True
        elif label == 'static_lines':
            mv.static_lines = obj
            self.need_redraw = True
        elif label == 'autorecenter':
            mv.autorecenter = obj
            self.need_redraw = True
        elif label == 'titlebar':
            assert isinstance(obj, str) or isinstance(obj, unicode)
            self.titlebar = obj
            glut.glutSetWindowTitle(obj)
        elif label == 'background_color':
            gl.glClearColor(obj[0], obj[1], obj[2], 1.0)
            self.need_redraw = True
        elif label == 'save_snapshot':  # redraws for itself
            assert isinstance(obj, str) or isinstance(obj, unicode)
            self.snapshot(obj)
        elif label == 'get_keypress':
            self.keypress_port = obj
        elif label == 'get_mouseclick':
            self.mouseclick_port = obj
        elif label == 'get_event':
            self.event_port = obj
        else:
            return False  # can't handle this request string
        return True  # handled the request string

    def checkQueue(self, unused_timer_id):  # pylint: disable=unused-argument
        glut.glutTimerFunc(20, self.checkQueue, 0)
        try:
            request = self.server.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                raise  # something wrong besides empty queue
            return  # empty queue, no problem
        if not request:
            return
        while request:
            task_completion_time = time.time()
            if not self.handle_request(request):
                raise Exception('Unknown command string: %s' % (request['label']))
            task_completion_time = time.time() - task_completion_time
            if 'port' in request:  # caller wants confirmation
                port = request['port']
                client = zmq.Context.instance().socket(zmq.PUSH)
                client.connect('tcp://127.0.0.1:%d' % (port))
                client.send_pyobj(task_completion_time)
            try:
                request = self.server.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise
                request = None
        if self.need_redraw:
            glut.glutPostRedisplay()

    # A general OpenGL initialization function.  Sets all of the initial parameters.
    def init_opengl(self):  # We call this right after our OpenGL window is created.
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # This Will Clear The Background Color To Black
        gl.glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
        gl.glDepthFunc(gl.GL_LEQUAL)  # The Type Of Depth Test To Do
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enables Depth Testing
        gl.glShadeModel(gl.GL_SMOOTH)
        # Really Nice Perspective Calculations:
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glEnable(gl.GL_NORMALIZE)  # important since we rescale the modelview matrix
        return True


def main(argv):
    if len(argv) > 2:
        # Start a remote window
        MeshViewerRemote(
            titlebar=argv[1], subwins_vert=int(argv[2]),
            subwins_horz=int(argv[3]), width=int(argv[4]), height=int(argv[5])
        )
    elif len(argv) == 2:
        # Just a test for opengl working
        try:
            from OpenGL.GLUT import glutInit
            glutInit()
        except Exception as e:  # pylint: disable=broad-except
            print >>sys.stderr, e
            print 'failure'
        else:
            print 'success'


if __name__ == '__main__':
    # Windows specific. See:
    # http://docs.python.org/2/library/multiprocessing.html#multiprocessing.freeze_support
    freeze_support()
    main(sys.argv)
