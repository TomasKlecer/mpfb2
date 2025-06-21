"""Operator for importing poses from asset library."""

import bpy
from bpy.props import StringProperty
from ....services import LogService
from ....services import ObjectService
from ....services import RigService
from ....services import AnimationService
from .... import ClassManager

_LOG = LogService.get_logger("assetlibrary.loadlibrarypose")


class MPFB_OT_Load_Library_Pose_Operator(bpy.types.Operator):
    """Destructively load a pose from a MH BVH file. WARNING: This will change the bone rolls of all bones, making further posing a bit unpredictable"""
    bl_idname = "mpfb.load_library_pose"
    bl_label = "Load Pose"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(name="filepath", description="Full path to asset", default="")
    object_type: StringProperty(name="object_type", description="type of the object", default="bvh")

    def execute(self, context):
        _LOG.debug("filepath", self.filepath)

        blender_object = context.active_object

        if blender_object.type != 'ARMATURE':
            blender_object = ObjectService.find_object_of_type_amongst_nearest_relatives(blender_object, "Skeleton")

        if not blender_object or blender_object.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object is not an armature")
            return {'CANCELLED'}

        if not ObjectService.object_is_skeleton(blender_object):
            self.report({'ERROR'}, "Active object is not identified as a skeleton")
            return {'CANCELLED'}

        skeleton = RigService.identify_rig(blender_object)

        if "default" not in skeleton:
            self.report({'ERROR'}, "BVH-style poses only work with default rig")
            return {'CANCELLED'}

        try:
            AnimationService.import_bvh_file_as_pose(blender_object, self.filepath)
            self.report({'INFO'}, "Pose loaded successfully")
        except Exception as e:
            _LOG.error("Failed to load pose", e)
            self.report({'ERROR'}, f"Failed to load pose: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Load_Library_Pose_Operator)
