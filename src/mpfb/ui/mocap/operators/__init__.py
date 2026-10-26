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
from .run_mocap_operator import (MPFB_OT_Mocap_Run_Operator, MPFB_OT_Mocap_Cancel_Run_Operator,
                                 MPFB_OT_Mocap_Open_Output_Operator, MPFB_OT_Mocap_Add_Video_Operator,
                                 MPFB_OT_Mocap_Remove_Video_Operator)
from .source_operators import (MPFB_OT_Mocap_Pick_Source, MPFB_OT_Mocap_Clear_Source, MPFB_OT_Mocap_Remove_Body_Source,
                               MPFB_OT_Mocap_Move_Body_Source, MPFB_OT_Mocap_Use_Last_Run)
from .face_animation_operator import MPFB_OT_Mocap_Create_Face_Animation_Operator

__all__ = [
    "MPFB_OT_Mocap_Install_Env_Operator",
    "MPFB_OT_Mocap_Cancel_Install_Operator",
    "MPFB_OT_Mocap_Open_Env_Dir_Operator",
    "MPFB_OT_Mocap_Run_Operator",
    "MPFB_OT_Mocap_Cancel_Run_Operator",
    "MPFB_OT_Mocap_Open_Output_Operator",
    "MPFB_OT_Mocap_Add_Video_Operator",
    "MPFB_OT_Mocap_Remove_Video_Operator",
    "MPFB_OT_Mocap_Pick_Source",
    "MPFB_OT_Mocap_Clear_Source",
    "MPFB_OT_Mocap_Remove_Body_Source",
    "MPFB_OT_Mocap_Move_Body_Source",
    "MPFB_OT_Mocap_Use_Last_Run",
    "MPFB_OT_Mocap_Create_Face_Animation_Operator"
]
