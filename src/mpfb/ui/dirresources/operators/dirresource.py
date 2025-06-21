"""Operator for opening dir browser"""

import bpy
from bpy.props import StringProperty
from ....services import LogService
from ....services import SystemService
from .... import ClassManager

_LOG = LogService.get_logger("dirresources.dirresource")

class MPFB_OT_Dir_Resource_Operator(bpy.types.Operator):
    """Open dir browser"""
    bl_idname = "mpfb.dir_resource"
    bl_label = "Open"
    bl_options = {'REGISTER'}

    path: StringProperty(name="path", description="Path to open", default="")

    def execute(self, context):
        SystemService.open_file_browser(self.path)
        return {'FINISHED'}

ClassManager.add_class(MPFB_OT_Dir_Resource_Operator)
