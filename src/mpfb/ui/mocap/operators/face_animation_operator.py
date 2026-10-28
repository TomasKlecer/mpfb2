# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         26.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operator for creating the ARKit face animation from the face source
# ------------------------------------------------------------------------------

from .... import ClassManager
from ....services.logservice import LogService
from ....services.objectservice import ObjectService
from ....services.mocapsourceservice import MocapSourceService, ROLE_FACE, ERROR
from ....services.mocapanimationservice import MocapAnimationService
from ...mpfboperator import MpfbOperator
from ..mocapproperties import read_face_settings
from ..sourceproperties import FACE_SOURCE, absolute

_LOG = LogService.get_logger("mocap.face_animation_operator")


def find_basemesh(context):
    if context.active_object is None:
        return None
    return ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")


class MPFB_OT_Mocap_Create_Face_Animation_Operator(MpfbOperator):
    """Create ARKit face animation and head rotation of selected character from given face sourcee"""
    bl_idname = "mpfb.mocap_create_face_animation"
    bl_label = "Create face animation"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if find_basemesh(context) is None:
            cls.poll_message_set("Select an MPFB2 character")
            return False
        if not absolute(getattr(context.scene, FACE_SOURCE, "")):
            cls.poll_message_set("Pick a face source first")
            return False
        return True

    def hardened_execute(self, context):
        basemesh = find_basemesh(context)
        path = absolute(getattr(context.scene, FACE_SOURCE))
        info = MocapSourceService.inspect(path)
        errors = [message for level, message in MocapSourceService.check_role(info, ROLE_FACE) if level == ERROR]
        if errors:
            self.report({'ERROR'}, errors[0])
            return {'CANCELLED'}

        try:
            result = MocapAnimationService.create_face_animation(basemesh, path, read_face_settings(context.scene),
                                                                 context.scene, info.get("input_name"))
        except (RuntimeError, ValueError) as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}

        message = "%s: %d shape keys, frames %d - %d" % (result["action"], result["animated"], result["frame_start"],
                                                        result["frame_end"])
        if result["children"]:
            message += ", %d fitted meshes follow" % result["children"]
        if result["head"]:
            message += ", head on %s" % ", ".join(result["head"]["bones"])
            if result["head"]["pushed_down"]:
                message += " (the action of the rig was moved to NLA track %s)" % result["head"]["pushed_down"]
        if result["head_skipped"]:
            message += ", no head rotation: " + result["head_skipped"]
        self.report({'WARNING'} if result["head_skipped"] else {'INFO'}, message)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Create_Face_Animation_Operator)
