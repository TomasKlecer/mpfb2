import bpy, json, math, os
from bpy.props import StringProperty
from ....services import LogService
from ....services import AnimationService
from ...mpfboperator import MpfbOperator
from .... import ClassManager
from bpy_extras.io_utils import ImportHelper

_LOG = LogService.get_logger("applypose.loadmhbvh")


class MPFB_OT_Load_MH_BVH_Operator(MpfbOperator, ImportHelper):
    """Destructively load a pose from a MH BVH file. WARNING: This will change the bone rolls of all bones, making further posing a bit unpredictable"""
    bl_idname = "mpfb.load_mhbvh_pose"
    bl_label = "Import MH BVH Pose"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default='*.bvh', options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        _LOG.enter()
        if context.object is None:
            return False
        if context.object is None or context.object.type != 'ARMATURE':
            return False
        return True

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        _LOG.enter()

        if context.object is None or context.object.type != 'ARMATURE':
            self.report({'ERROR'}, "Must have armature as active object")
            return {'FINISHED'}

        armature_object = context.object

        _LOG.debug("filepath", self.filepath)

        AnimationService.import_bvh_file_as_pose(armature_object, self.filepath)

        self.report({'INFO'}, "The pose was destructively loaded from " + self.filepath)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Load_MH_BVH_Operator)
