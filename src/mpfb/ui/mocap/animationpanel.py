# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  motion capture subpanel for creating animation from the model outputs
# ------------------------------------------------------------------------------

import bpy

from ... import ClassManager
from ...services.logservice import LogService
from ...services.uiservice import UiService
from ...services.mocapsourceservice import MocapSourceService, ROLE_FACE, ROLE_BODY, ROLE_AUDIO, ERROR, KIND_LIVELINK
from ...services.mocapanimationservice import MocapAnimationService
from ...ui.abstractpanel import Abstract_Panel
from .mocappanel import multiline
from .mocapproperties import MOCAP_PROPERTIES, speech_from_audio
from .operators.face_animation_operator import find_basemesh
from .sourceproperties import FACE_SOURCE, AUDIO_SOURCE, BODY_SOURCES, BODY_INDEX, absolute, main_body_index

_LOG = LogService.get_logger("ui.animationpanel")

_MESSAGE_ICONS = {ERROR: "ERROR"}


def _draw_messages(layout, messages):
    for level, message in messages:
        layout.label(text=message, icon=_MESSAGE_ICONS.get(level, "INFO"))


class MPFB_UL_Mocap_Body_Sources(bpy.types.UIList):
    """Body sources: checkbox enabled, star at the main source, name of the input and the run, pose detection percent."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        path = absolute(item.path)
        if not path:
            row.label(text="(no file)", icon="QUESTION")
            return
        info = MocapSourceService.inspect(path)
        is_main = index == main_body_index(data)
        problems = MocapSourceService.check_role(info, ROLE_BODY)
        has_error = any(level == ERROR for level, _message in problems)
        row.label(text=MocapSourceService.describe(info),
                  icon="ERROR" if has_error else ("SOLO_ON" if is_main else "BLANK1"))
        rate = info.get("detection", {}).get("pose")
        if rate is not None and not has_error:
            row.label(text="%d %%" % round(rate * 100))


class MPFB_PT_Mocap_Animation_Panel(Abstract_Panel):
    """Turns the saved outputs of the models (npz files) into shape key and bone animation."""
    bl_label = "Create Animation"
    bl_category = UiService.get_value("MOCAPCATEGORY")
    bl_parent_id = "MPFB_PT_Mocap_Panel"
    bl_options = set()
    bl_order = 2

    def _single_source(self, layout, scene, title, property_name, role):
        """Box with one source: path field, browse and clear buttons, file and its eventual problems."""
        box = self.create_box(layout, title)
        row = box.row(align=True)
        row.prop(scene, property_name, text="")
        row.operator("mpfb.mocap_pick_source", text="", icon="FILEBROWSER").role = role
        row.operator("mpfb.mocap_clear_source", text="", icon="X").role = role
        path = absolute(getattr(scene, property_name))
        if not path:
            box.label(text="Not used")
            return None
        info = MocapSourceService.inspect(path)
        column = box.column(align=True)
        column.scale_y = 0.8
        if info.get("exists") and not info.get("error"):
            column.label(text=MocapSourceService.describe(info))
            column.label(text=MocapSourceService.details(info))
        _draw_messages(column, MocapSourceService.check_role(info, role))
        return info

    def _body_sources(self, layout, scene):
        box = self.create_box(layout, "Body")
        row = box.row()
        row.template_list("MPFB_UL_Mocap_Body_Sources", "", scene, BODY_SOURCES, scene, BODY_INDEX, rows=3)
        buttons = row.column(align=True)
        buttons.operator("mpfb.mocap_pick_source", text="", icon="ADD").role = ROLE_BODY
        buttons.operator("mpfb.mocap_remove_body_source", text="", icon="REMOVE")
        buttons.separator()
        buttons.operator("mpfb.mocap_move_body_source", text="", icon="TRIA_UP").direction = "UP"
        buttons.operator("mpfb.mocap_move_body_source", text="", icon="TRIA_DOWN").direction = "DOWN"

        sources = getattr(scene, BODY_SOURCES)
        infos = [MocapSourceService.inspect(absolute(source.path)) for source in sources
                 if source.enabled and absolute(source.path)]
        index = getattr(scene, BODY_INDEX)
        if 0 <= index < len(sources):
            source = sources[index]
            column = box.column(align=True)
            column.scale_y = 0.8
            column.prop(source, "path", text="")
            path = absolute(source.path)
            if path:
                info = MocapSourceService.inspect(path)
                if info.get("exists") and not info.get("error"):
                    column.label(text=MocapSourceService.details(info))
                _draw_messages(column, MocapSourceService.check_role(info, ROLE_BODY))
        if len(infos) > 1:
            box.label(text="The starred source is the main one (timeline, reference camera).", icon="SOLO_ON")
        elif not len(sources):
            box.label(text="Add one or more *_landmarks.npz with +")
        return infos

    def _face_animation(self, layout, scene, context, face):
        """Settings of the ARKit face animation and the button to create it."""
        box = self.create_box(layout, "Face animation")
        MOCAP_PROPERTIES.draw_properties(scene, box, ["face_smoothing", "face_gain"], slider=True)
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["face_use_neutral"])
        neutral = row.row(align=True)
        neutral.enabled = bool(MOCAP_PROPERTIES.get_value("face_use_neutral", False, entity_reference=scene))
        MOCAP_PROPERTIES.draw_properties(scene, neutral, ["face_neutral_time"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["face_mirror"])
        if absolute(getattr(scene, AUDIO_SOURCE, "")):
            MOCAP_PROPERTIES.draw_properties(scene, box, ["face_keep_speech"])
            if speech_from_audio(scene):
                box.label(text="Jaw, lips and tongue are left to the lipsync.", icon="INFO")
        if face is not None and face.get("kind") == KIND_LIVELINK and MOCAP_PROPERTIES.get_value("face_mirror", False, entity_reference=scene):
            box.label(text="Live Link Face already uses the actor's left/right.", icon="INFO")

        basemesh = find_basemesh(context)
        if basemesh is None:
            box.label(text="Select an MPFB2 character.", icon="INFO")
        elif face is None:
            box.label(text="Pick a face source above.", icon="INFO")
        else:
            box.label(text="Starts at frame %d on %s" % (scene.frame_current, basemesh.name))
        row = box.row()
        row.scale_y = 1.5
        row.operator("mpfb.mocap_create_face_animation", icon="SHAPEKEY_DATA")

    def _lipsync(self, layout, scene, context, audio):
        """Settings of the lipsync and the button to create it."""
        box = self.create_box(layout, "Lipsync")
        MOCAP_PROPERTIES.draw_properties(scene, box, ["lipsync_strength"], slider=True)
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["lipsync_lead", "lipsync_transition"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["lipsync_add_sound"])
        basemesh = find_basemesh(context)
        if basemesh is None:
            box.label(text="Select an MPFB2 character.", icon="INFO")
        elif audio is None:
            box.label(text="Pick an audio source above.", icon="INFO")
        else:
            box.label(text="Starts at frame %d on %s" % (scene.frame_current, basemesh.name))
        row = box.row()
        row.scale_y = 1.5
        row.operator("mpfb.mocap_create_lipsync", icon="SPEAKER")

    def _layers(self, layout, context):
        """The NLA layers of the selected character and the bake button."""
        box = self.create_box(layout, "Layers")
        MOCAP_PROPERTIES.draw_properties(context.scene, box, ["face_include_children", "face_extend_scene"])
        basemesh = find_basemesh(context)
        tracks = MocapAnimationService.layer_tracks(basemesh) if basemesh is not None else []
        if not tracks:
            box.label(text="Face and lipsync are kept as separate layers (NLA tracks).")
            return
        column = box.column(align=True)
        for track in tracks:
            for strip in track.strips:
                column.label(text="%s: %s (%d - %d)%s" % (track.name, strip.action.name if strip.action else "-",
                                                         strip.frame_start, strip.frame_end, ", muted" if track.mute else ""),
                             icon="NLA")
        animation_data = basemesh.data.shape_keys.animation_data
        if animation_data and animation_data.action:
            box.label(text="Active action: " + animation_data.action.name, icon="ACTION")
        row = box.row()
        row.scale_y = 1.2
        row.operator("mpfb.mocap_bake_layers", icon="RENDER_ANIMATION")

    def draw(self, context):
        _LOG.enter()
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.label(text="Sources (saved outputs of Run Models)")
        row.operator("mpfb.mocap_use_last_run", text="Use last run", icon="FILE_REFRESH")
        face = self._single_source(layout, scene, "Face", FACE_SOURCE, ROLE_FACE)
        bodies = self._body_sources(layout, scene)
        audio = self._single_source(layout, scene, "Audio", AUDIO_SOURCE, ROLE_AUDIO)

        problems = MocapSourceService.check_together(face, bodies, audio)
        if problems:
            box = self.create_box(layout, "Checks")
            _draw_messages(box, problems)

        self._face_animation(layout, scene, context, face)
        self._lipsync(layout, scene, context, audio)
        self._layers(layout, context)
        multiline(layout, ["Body animation will come here."])


ClassManager.add_class(MPFB_UL_Mocap_Body_Sources)
ClassManager.add_class(MPFB_PT_Mocap_Animation_Panel)
