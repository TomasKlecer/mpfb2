from ....services import LogService
from ....services import LocationService
from ....services import MaterialService
from ....services import ObjectService
from ....services import AnimationService
from ....services import RigService
from .... import ClassManager
from ...mpfboperator import MpfbOperator
import bpy, json, math, os
from bpy.types import StringProperty
from bpy_extras.io_utils import ExportHelper

_LOG = LogService.get_logger("animops.repeatanim")

class MPFB_OT_Make_Cyclic_Operator(MpfbOperator):
    """Make animation cyclic by adding fcurve modifiers. WARNING: This will not work if the animation already has fcurve modifiers."""
    bl_idname = "mpfb.make_cyclic"
    bl_label = "Make cyclic"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        _LOG.enter()
        if context.object is None or context.object.type != 'ARMATURE':
            return False
        return True

    def hardened_execute(self, context):
        _LOG.enter()

        if context.object is None or context.object.type != 'ARMATURE':
            self.report({'ERROR'}, "Must have armature as active object")
            return {'CANCELLED'}

        armature_object = context.object

        from ...animops.animopspanel import ANIMOPS_PROPERTIES

        shiftroot = ANIMOPS_PROPERTIES.get_value("shiftroot", entity_reference=context.scene)
        rootbone = ANIMOPS_PROPERTIES.get_value('rootbone', entity_reference=context.scene)

        bone_name = None
        if shiftroot and rootbone:
            bone_name = rootbone

        bpy.ops.object.mode_set(mode='POSE', toggle=False)

        AnimationService.make_cyclic(armature_object, bone_name)

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        self.report({'INFO'}, "Done")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Make_Cyclic_Operator)
