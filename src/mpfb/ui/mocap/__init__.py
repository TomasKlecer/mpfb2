# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  init file for motion capture
# ------------------------------------------------------------------------------
"""Motion capture from video module."""

from ...services import LogService as _LogService
_LOG = _LogService.get_logger("mocap.init")
_LOG.trace("initializing motion capture module")

from .mocappanel import MPFB_PT_Mocap_Panel
from .operators import *

__all__ = [
    "MPFB_PT_Mocap_Panel",
    "MPFB_OT_Mocap_Install_Env_Operator",
    "MPFB_OT_Mocap_Cancel_Install_Operator",
    "MPFB_OT_Mocap_Open_Env_Dir_Operator"
]
