# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         26.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operators for the lipsync from the audio source
# ------------------------------------------------------------------------------

from .... import ClassManager
from ....services.logservice import LogService
from ....services.mocapsourceservice import MocapSourceService, ROLE_AUDIO, ERROR
from ....services.mocapanimationservice import MocapAnimationService
from ...mpfboperator import MpfbOperator
from ..mocapproperties import read_lipsync_settings
from ..sourceproperties import AUDIO_SOURCE, absolute
from .face_animation_operator import find_basemesh

_LOG = LogService.get_logger("mocap.lipsync_operator")


class MPFB_OT_Mocap_Create_Lipsync_Operator(MpfbOperator):
    """Create lipsync (visemes02) of selected character from the audio source, starting at the current frame"""
    bl_idname = "mpfb.mocap_create_lipsync"
    bl_label = "Create lipsync"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if find_basemesh(context) is None:
            cls.poll_message_set("Select an MPFB2 character")
            return False
        if not absolute(getattr(context.scene, AUDIO_SOURCE, "")):
            cls.poll_message_set("Pick an audio source first")
            return False
        return True

    def hardened_execute(self, context):
        basemesh = find_basemesh(context)
        path = absolute(getattr(context.scene, AUDIO_SOURCE))
        info = MocapSourceService.inspect(path)
        errors = [message for level, message in MocapSourceService.check_role(info, ROLE_AUDIO) if level == ERROR]
        if errors:
            self.report({'ERROR'}, errors[0])
            return {'CANCELLED'}

        try:
            result = MocapAnimationService.create_lipsync(basemesh, path, read_lipsync_settings(context.scene),
                                                          context.scene, info)
        except (RuntimeError, ValueError) as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}

        message = "%s: %d phonemes, frames %d - %d" % (result["action"], result["phonemes"], result["frame_start"],
                                                      result["frame_end"])
        if result["unknown"]:
            message += ", no viseme for " + " ".join(result["unknown"])
        self.report({'WARNING'} if result["unknown"] else {'INFO'}, message)
        return {'FINISHED'}


class MPFB_OT_Mocap_Bake_Layers_Operator(MpfbOperator):
    """Bake face and lipsync layers of selected character into one action (the layers are muted, not removed)"""
    bl_idname = "mpfb.mocap_bake_layers"
    bl_label = "Bake to one action"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        basemesh = find_basemesh(context)
        if basemesh is None:
            cls.poll_message_set("Select an MPFB2 character")
            return False
        if not MocapAnimationService.layer_tracks(basemesh):
            cls.poll_message_set("Create a face or lipsync animation first")
            return False
        return True

    def hardened_execute(self, context):
        try:
            result = MocapAnimationService.bake_layers(find_basemesh(context))
        except RuntimeError as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}
        self.report({'INFO'}, "%s: %d shape keys, frames %d - %d" % (result["action"], result["channels"],
                                                                     result["frame_start"], result["frame_end"]))
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Create_Lipsync_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Bake_Layers_Operator)
