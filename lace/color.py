# pylint: disable=attribute-defined-outside-init
# http://matplotlib.org/examples/color/colormaps_reference.html
DEFAULT_COLORMAP = 'jet'

def colors_like(color, arr, colormap=DEFAULT_COLORMAP):
    '''
    Given an array of size NxM (usually Nx3), we accept color in the following ways:
    - A string color name. The accepted names are roughly what's in X11's rgb.txt
    - An explicit rgb triple, in (3, ), (3, 1), or (1, 3) shape
    - A list of values (N, ), (N, 1), or (1, N) that are put through a colormap to get per vertex color
    - An array of colors (N, 3) or (3, N)

    There is a potential for conflict here if N == 3. In that case we assume a value is an rgb triple,
    not a colormap index. This is a sort of degenerate case, as a mesh with three verticies is just a single
    triangle and not something we ever actually use in practice.
    '''
    import six
    import numpy as np
    from blmath.numerics import is_empty_arraylike
    if is_empty_arraylike(color):
        return None
    if isinstance(color, six.string_types):
        from lace.color_names import name_to_rgb
        color = name_to_rgb[color]
    elif isinstance(color, list):
        color = np.array(color)
    color = np.squeeze(color)
    num_verts = arr.shape[0]
    if color.ndim == 1:
        if color.shape[0] == 3: # rgb triple
            return np.ones((num_verts, 3)) * np.array([color])
        else:
            from matplotlib import cm
            return np.ones((num_verts, 3)) * cm.get_cmap(colormap)(color.flatten())[:, :3]
    elif color.ndim == 2:
        if color.shape[1] == num_verts:
            color = color.T
        return np.ones((num_verts, 3)) * color
    else:
        raise ValueError("Colors must be specified as one or two dimensions")


class MeshMixin(object):

    def set_vertex_colors(self, vc, vertex_indices=None):
        import numpy as np
        if vertex_indices != None:
            if self.vc is None:
                self.vc = np.zeros_like(self.v)
            self.vc[vertex_indices] = colors_like(vc, self.v[vertex_indices])
        else:
            import warnings
            warnings.warn("Mesh.set_vertex_colors without vertex_indices is deprecated in favor of just setting mesh.vc directly", DeprecationWarning)
            self.vc = vc

    def set_vertex_colors_from_weights(self, weights, scale_to_range_1=True, color=DEFAULT_COLORMAP):
        import numpy as np
        if scale_to_range_1:
            from blmath.numerics import scale_to_range
            weights = scale_to_range(weights.flatten(), 0.0, 1.0)
        if color != None:
            self.vc = colors_like(weights, self.v, colormap=color)
        else:
            self.vc = np.tile(np.reshape(weights, (len(weights), 1)), (1, 3))

    def scale_vertex_colors(self, weights, w_min=0.0, w_max=1.0):
        from blmath.numerics import scale_to_range
        weights = scale_to_range(weights.flatten(), w_min, w_max)
        self.vc = weights.reshape(-1, 1) * self.vc

    def set_face_colors(self, fc):
        import warnings
        warnings.warn("Mesh.set_face_colors is deprecated in favor of just setting mesh.fc directly", DeprecationWarning)
        self.fc = fc

    def set_face_colors_from_weights(self, weights, scale_to_range_1=True, color=DEFAULT_COLORMAP):
        import numpy as np
        if scale_to_range_1:
            from blmath.numerics import scale_to_range
            weights = scale_to_range(weights.flatten(), 0.0, 1.0)
        if color != None:
            self.fc = colors_like(weights, self.f, colormap=color)
        else:
            self.fc = np.tile(np.reshape(weights, (len(weights), 1)), (1, 3))

    def color_by_height(self, axis=1, threshold=None, color=DEFAULT_COLORMAP): # pylint: disable=unused-argument
        '''
        Color each vertex by its height above the floor point.

        axis: The axis to use. 0 means x, 1 means y, 2 means z.
        threshold: The top of the range. Points at and above this height will
          be the same color.

        '''
        import numpy as np

        heights = self.v[:, axis] - self.floor_point[axis]

        if threshold:
            # height == 0  ->  saturation = 0.
            # height == threshold  ->  saturation = 1.
            color_weights = np.minimum(heights / threshold, 1.)
            color_weights = color_weights * color_weights
            self.set_vertex_colors_from_weights(color_weights, scale_to_range_1=False)
        else:
            self.set_vertex_colors_from_weights(heights)

    def color_as_heatmap(self, weights, max_weight=1.0, color=DEFAULT_COLORMAP):
        '''
        Truncate weights to the range [0, max_weight] and rescale, as you would
        want for a heatmap with known scale.
        '''
        import numpy as np
        adjusted_weights = np.clip(weights, 0., max_weight) / max_weight
        self.vc = colors_like(adjusted_weights, self.v, colormap=color)
