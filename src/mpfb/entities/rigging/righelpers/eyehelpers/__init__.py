"""This module hierarchy provides utility classes for adding and removing helper
bones (for example IK targets) to the eyes section of a makehuman rig."""

from .....services import LogService

_LOG = LogService.get_logger("eyehelpers.init")
_LOG.trace("initializing eyehelpers module")

from .eyehelpers import EyeHelpers
from .defaulteyehelpers import DefaultEyeHelpers

__all__ = ["EyeHelpers", "DefaultEyeHelpers"]
