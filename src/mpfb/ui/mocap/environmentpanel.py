# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  motion capture subpanel with the python environment the models run in
# ------------------------------------------------------------------------------

import bpy

from ... import ClassManager
from ...services.logservice import LogService
from ...services.uiservice import UiService
from ...services.mocapenvservice import MocapEnvService, InstallJob
from ...ui.abstractpanel import Abstract_Panel
from .mocappanel import format_seconds, multiline, MAX_LINE_LENGTH

_LOG = LogService.get_logger("ui.environmentpanel")


class MPFB_PT_Mocap_Environment_Panel(Abstract_Panel):
    """Installing and checking the Python environment the models run in."""
    bl_label = "Environment"
    bl_category = UiService.get_value("MOCAPCATEGORY")
    bl_parent_id = "MPFB_PT_Mocap_Panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 3

    def _draw_running(self, layout, job):
        box = self.create_box(layout, "Installing motion capture environment")
        box.label(text="Step %d/%d: %s" % (job.step_index, job.step_count, job.step_label), icon="TIME")
        box.label(text="Elapsed: " + format_seconds(job.elapsed_seconds))
        if job.last_line:
            box.label(text=job.last_line[:MAX_LINE_LENGTH])
        if job.step_label.startswith("Installing packages"):
            box.label(text="Downloading packages can take a long time.", icon="INFO")
        row = box.row()
        row.operator("mpfb.mocap_cancel_install", icon="CANCEL")
        row.operator("mpfb.mocap_open_env_dir", text="Open log folder", icon="FILE_FOLDER")

    def _draw_ready(self, layout, env_dir):
        column = layout.column()
        column.label(text="Motion capture environment is ready", icon="CHECKMARK")
        marker = MocapEnvService.read_marker(env_dir) or {}
        versions = marker.get("versions") or {}
        if versions.get("cuda"):
            column.label(text="Torch device: CUDA (%s)" % (versions.get("cuda_device") or "GPU"))
        else:
            column.label(text="Torch device: CPU")
        if versions.get("mediapipe"):
            column.label(text="MediaPipe %s, Torch %s" % (versions.get("mediapipe"), versions.get("torch")))
        column.operator("mpfb.mocap_open_env_dir", icon="FILE_FOLDER")

        self._draw_missing_models(layout)

    def _draw_missing_models(self, layout):
        missing = MocapEnvService.get_missing_models()
        if missing:
            box = self.create_box(layout, "Missing models")
            box.label(text="These models are missing in data/ai_models:", icon="ERROR")
            for model_id in missing:
                box.label(text=model_id)

    def _draw_last_failure(self, layout, job, env_dir):
        if job.status not in (InstallJob.FAILED, InstallJob.CANCELLED) or job.env_dir != env_dir:
            return
        box = self.create_box(layout, "Last install " + ("was cancelled" if job.status == InstallJob.CANCELLED else "failed"))
        if job.status == InstallJob.FAILED:
            box.label(text=job.error[:MAX_LINE_LENGTH], icon="ERROR")
            box.label(text="See install.log in the environment folder.")
        box.operator("mpfb.mocap_open_env_dir", text="Open log folder", icon="FILE_FOLDER")

    def _draw_setup(self, layout, job, env_dir):
        self._draw_last_failure(layout, job, env_dir)

        box = self.create_box(layout, "Setup")
        multiline(box, [
            "The models run in their own Python,",
            "in environment managed by uv.",
            "Everything runs locally, no video or",
            "audio ever leaves your computer.",
            "The install downloads Python and packages",
            "(about 1 GB, or up to 5 GB with CUDA)."])

        prefs = MocapEnvService.get_preferences()
        if prefs is not None:
            column = box.column()
            column.label(text="Environment folder:")
            try:
                column.prop(prefs, "mocap_env_dir", text="", placeholder=MocapEnvService.get_default_env_dir())
            except TypeError:
                # placeholder is not known by every blender version
                column.prop(prefs, "mocap_env_dir", text="")
                column.label(text="Empty means " + MocapEnvService.get_default_env_dir())
            column.prop(prefs, "mocap_use_own_uv")
            if prefs.mocap_use_own_uv:
                column.prop(prefs, "mocap_uv_binary", text="")

        if not bpy.app.online_access:
            warning = box.column()
            warning.label(text="Online access is disabled.", icon="ERROR")
            warning.label(text="Enable it in Preferences > System > Network.")

        self._draw_missing_models(layout)

        row = box.row()
        row.scale_y = 1.5
        row.operator("mpfb.mocap_install_env", icon="IMPORT")

    def draw(self, context):
        _LOG.enter()
        layout = self.layout
        job = MocapEnvService.get_job()
        if job.is_running:
            self._draw_running(layout, job)
            return

        env_dir = MocapEnvService.get_env_dir()
        if MocapEnvService.is_env_installed(env_dir):
            self._draw_ready(layout, env_dir)
        else:
            self._draw_setup(layout, job, env_dir)


ClassManager.add_class(MPFB_PT_Mocap_Environment_Panel)
