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
from .runmodelspanel import MPFB_PT_Mocap_Run_Models_Panel
from .animationpanel import MPFB_PT_Mocap_Animation_Panel, MPFB_UL_Mocap_Body_Sources
from .environmentpanel import MPFB_PT_Mocap_Environment_Panel
from .sourceproperties import (MPFB_PG_Mocap_Body_Source, MPFB_PG_Mocap_Video_Input, register_scene_properties,
                               unregister_scene_properties)
from .operators import *

__all__ = [
    "MPFB_PT_Mocap_Panel",
    "MPFB_PT_Mocap_Run_Models_Panel",
    "MPFB_PT_Mocap_Animation_Panel",
    "MPFB_PT_Mocap_Environment_Panel",
    "MPFB_UL_Mocap_Body_Sources",
    "MPFB_PG_Mocap_Body_Source",
    "MPFB_PG_Mocap_Video_Input",
    "register_scene_properties",
    "unregister_scene_properties",
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
    "MPFB_OT_Mocap_Create_Face_Animation_Operator",
    "MPFB_OT_Mocap_Create_Lipsync_Operator",
    "MPFB_OT_Mocap_Bake_Layers_Operator"
]
