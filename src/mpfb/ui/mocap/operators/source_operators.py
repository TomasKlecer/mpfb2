# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operators for picking the sources of the Create Animation panel
# ------------------------------------------------------------------------------

import os

import bpy
from bpy.props import CollectionProperty, EnumProperty, StringProperty

from .... import ClassManager
from ....services.logservice import LogService
from ....services.mocapsourceservice import MocapSourceService, ROLE_FILTERS, ROLE_FACE, ROLE_BODY, ROLE_AUDIO
from ...mpfboperator import MpfbOperator
from ..sourceproperties import FACE_SOURCE, AUDIO_SOURCE, BODY_SOURCES, BODY_INDEX, absolute, fill_sources

_LOG = LogService.get_logger("mocap.source_operators")

_ROLES = [(ROLE_FACE, "Face", "Output used for the face"),
          (ROLE_BODY, "Body", "Output used for the body (added to the list)"),
          (ROLE_AUDIO, "Audio", "Output used for the mouth")]
_SINGLE_SOURCE_PROPERTY = {ROLE_FACE: FACE_SOURCE, ROLE_AUDIO: AUDIO_SOURCE}


class MPFB_OT_Mocap_Pick_Source(MpfbOperator):
    """Picks a saved output of the models. The file browser opens in the folder with the runs and shows only usable files."""
    bl_idname = "mpfb.mocap_pick_source"
    bl_label = "Pick source"
    bl_options = {'REGISTER', 'UNDO'}

    role: EnumProperty(items=_ROLES, name="Role", default=ROLE_BODY)
    filepath: StringProperty(subtype='FILE_PATH')
    directory: StringProperty(subtype='DIR_PATH')
    files: CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.npz", options={'HIDDEN'})

    def get_logger(self):
        return _LOG

    def _start_dir(self, context) -> str:
        """Folder of the current source if there is one, otherwise the latest run, otherwise the runs folder."""
        current = ""
        if self.role in _SINGLE_SOURCE_PROPERTY:
            current = absolute(getattr(context.scene, _SINGLE_SOURCE_PROPERTY[self.role]))
        else:
            sources = getattr(context.scene, BODY_SOURCES)
            if len(sources):
                current = absolute(sources[min(getattr(context.scene, BODY_INDEX), len(sources) - 1)].path)
        if current and os.path.isdir(os.path.dirname(current)):
            return os.path.dirname(current)
        return MocapSourceService.get_latest_run_dir() or MocapSourceService.get_runs_dir()

    def invoke(self, context, event):
        self.filter_glob = ROLE_FILTERS[self.role]
        self.filepath = os.path.join(self._start_dir(context), "")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def hardened_execute(self, context):
        scene = context.scene
        paths = [os.path.join(self.directory, item.name) for item in self.files if item.name]
        if not paths and self.filepath and not os.path.isdir(self.filepath):
            paths = [self.filepath]
        if not paths:
            self.report({'WARNING'}, "No file was selected")
            return {'CANCELLED'}

        if self.role in _SINGLE_SOURCE_PROPERTY:
            setattr(scene, _SINGLE_SOURCE_PROPERTY[self.role], paths[0])
            return {'FINISHED'}

        sources = getattr(scene, BODY_SOURCES)
        known = {absolute(source.path) for source in sources}
        added = 0
        for path in paths:
            if os.path.normpath(path) in known:
                continue
            source = sources.add()
            source.path = path
            source.enabled = True
            added += 1
        if added:
            setattr(scene, BODY_INDEX, len(sources) - 1)
        else:
            self.report({'INFO'}, "The selected files are already in the list")
        return {'FINISHED'}


class MPFB_OT_Mocap_Clear_Source(MpfbOperator):
    """Remove the face or audio source"""
    bl_idname = "mpfb.mocap_clear_source"
    bl_label = "Clear source"
    bl_options = {'REGISTER', 'UNDO'}

    role: EnumProperty(items=[item for item in _ROLES if item[0] != ROLE_BODY], name="Role", default=ROLE_FACE)

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        setattr(context.scene, _SINGLE_SOURCE_PROPERTY[self.role], "")
        return {'FINISHED'}


class MPFB_OT_Mocap_Remove_Body_Source(MpfbOperator):
    """Removed the selected body source from the list."""
    bl_idname = "mpfb.mocap_remove_body_source"
    bl_label = "Remove body source"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(getattr(context.scene, BODY_SOURCES, [])) > 0

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        scene = context.scene
        sources = getattr(scene, BODY_SOURCES)
        index = getattr(scene, BODY_INDEX)
        if 0 <= index < len(sources):
            sources.remove(index)
            setattr(scene, BODY_INDEX, max(0, min(index, len(sources) - 1)))
        return {'FINISHED'}


class MPFB_OT_Mocap_Move_Body_Source(MpfbOperator):
    """Moves the selected body source up or down. First enabled source is the main one."""
    bl_idname = "mpfb.mocap_move_body_source"
    bl_label = "Move body source"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")], default="UP")

    @classmethod
    def poll(cls, context):
        return len(getattr(context.scene, BODY_SOURCES, [])) > 1

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        scene = context.scene
        sources = getattr(scene, BODY_SOURCES)
        index = getattr(scene, BODY_INDEX)
        target = index - 1 if self.direction == "UP" else index + 1
        if 0 <= index < len(sources) and 0 <= target < len(sources):
            sources.move(index, target)
            setattr(scene, BODY_INDEX, target)
        return {'FINISHED'}


class MPFB_OT_Mocap_Use_Last_Run(MpfbOperator):
    """Fills sources with the outputs of the last run of Run Models."""
    bl_idname = "mpfb.mocap_use_last_run"
    bl_label = "Use last run"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if MocapSourceService.get_latest_run_dir() is None:
            cls.poll_message_set("There is no run yet, run the models in Run Models first")
            return False
        return True

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        run_dir = MocapSourceService.get_latest_run_dir()
        filled = fill_sources(context.scene, MocapSourceService.sources_from_run(run_dir))
        if not filled:
            self.report({'WARNING'}, "The last run has no usable outputs: " + os.path.basename(run_dir))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sources filled from run %s (%s)" % (os.path.basename(run_dir), ", ".join(filled)))
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Pick_Source)
ClassManager.add_class(MPFB_OT_Mocap_Use_Last_Run)
ClassManager.add_class(MPFB_OT_Mocap_Clear_Source)
ClassManager.add_class(MPFB_OT_Mocap_Remove_Body_Source)
ClassManager.add_class(MPFB_OT_Mocap_Move_Body_Source)
