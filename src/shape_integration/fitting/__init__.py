"""Pure sparse-observation fitting; no VLM, crop selection or scene mutation."""
from .adapter import fit_object, transform_fit
from .contracts import ObservationError, observation_schema, validate_observation
from .frames import FitFrame
from .primitives import FitError, fit_boundary

__all__ = ['fit_object', 'transform_fit', 'FitFrame', 'FitError', 'fit_boundary',
           'ObservationError', 'observation_schema', 'validate_observation']
