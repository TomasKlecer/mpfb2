# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  init file for motion capture operators
# ------------------------------------------------------------------------------
"""Operators for motion capture"""

from ....services import LogService as _LogService
_LOG = _LogService.get_logger("mocap.operators")
_LOG.trace("initializing motion capture operators")

from .install_env_operator import (MPFB_OT_Mocap_Install_Env_Operator, MPFB_OT_Mocap_Cancel_Install_Operator,
                                   MPFB_OT_Mocap_Open_Env_Dir_Operator)

__all__ = [
    "MPFB_OT_Mocap_Install_Env_Operator",
    "MPFB_OT_Mocap_Cancel_Install_Operator",
    "MPFB_OT_Mocap_Open_Env_Dir_Operator"
]
