import abc
from collections import OrderedDict


class OptionsSchema(object):
    """
    Base class for defining a static set of options with default values.

    A subclass defines an options schema and sets default values in
    `set_defaults`. Option overrides can be specified on initialization.
    Instance option values can later be modified with `update`, but the set
    of possible options is fixed by the definition.

    Example usage
    -------------
    >>> from bodylabs.util.options import OptionsSchema
    >>> class DeliSandwichOptions(OptionsSchema):
    ...     def set_defaults(self):
    ...         self.bread = 'wheat'
    ...         self.meat_type = None
    ...         self.dressing = 'spicy mayo'


    >>> options = DeliSandwichOptions(bread='pumpernickel')

    >>> print options.bread
    'pumpernickel'

    >>> print options.meat_type

    >>> print options.size
    AttributeError

    >>> options.update(bread='rye', meat_type='ham')

    >>> print options.meat_type
    'ham'

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        # Bypass __setattr__.
        self.__dict__.update({
            '_options': OrderedDict(),
            '_frozen': False,
        })

        self.set_defaults()

        self.__dict__['_frozen'] = True

        self.update(**kwargs)

    @property
    def option_names(self):
        return self._options.keys()

    @abc.abstractmethod
    def set_defaults(self):
        pass

    def update(self, **kwargs):
        for attr, value in kwargs.iteritems():
            if attr not in self._options:
                raise ValueError('{} is not a valid option'.format(attr))
            self._options[attr] = value

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            if attr not in self._options:
                raise AttributeError('{} is not a valid option'.format(attr))
            return self._options[attr]

    def __setattr__(self, attr, value):
        if attr not in self._options and self._frozen:
            raise AttributeError(
                'Option set can not be modified after initialization')

        self._options[attr] = value


class ScanAlignmentOptions(OptionsSchema):
    """Wraps options for a single stage of scan alignment.
    TODO: This is a temporary solution to deal with the unwieldy signature of
    `align_scan`. There are only a handful on concepts that we typically use
    when setting up an alignment optimization (e.g. s2m, m2s, edge coupling,
    regularization). These can be consolidated across scan, foot, head, and
    Realsense alignments. Maybe as mutators on an `OptimizationContext`?
    A downside of this single configuration object is that it may not be clear
    exactly what the configuration is for a given call to `align_scan`.
    One option is to create a new instance for each call rather than updating
    a single instance. It depends whether there is a lot of shared
    configuration between calls, where it might be better to emphasize how
    little changed (i.e. but updating one or two options before a new call).
    """
    def set_defaults(self):
        # Robustness term for scan-to-mesh.
        self.s2m_sigma = None
        # Point-specific weights for s2m objective.
        self.scan_weights = None

        # Global mesh-to-scan weight.
        self.m2s_weight = None

        # Global model coupling multiplier.
        self.model_coupling_weight = None
        # Vertex specific coupling weights.
        self.vertex_coupling_weights = None

        # Weight on shape regularization.
        self.shape_prior = 0.001
        # Weight on pose regularization.
        self.pose_prior = 0.001
        # Model parameters that should be fixed during the optimization.
        # Valid options are 'pose', 'betas', and 'trans'.
        self.fixed_model_params = []

        # Optimizes translation and rotation of partial scans in the case that
        # the input scan is a `MergedScan`.
        self.calibrate_scan = False

        # Residual tolerance for sparse solver.
        self.solver_tol = 1e-6
        # e_3 stopping criterion for dogleg optimization.
        self.min_inc_improvement = None

        # Head refinement.
        self.head_coupling_weight = 1.
