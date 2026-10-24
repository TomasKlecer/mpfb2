# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  UI panel for the motion capture module
# ------------------------------------------------------------------------------

import bpy

from ... import ClassManager
from ...services.logservice import LogService
from ...services.uiservice import UiService
from ...services.mocapenvservice import MocapEnvService, InstallJob
from ...services.mocaprunservice import MocapRunService, RunJob, TaskState
from ...ui.abstractpanel import Abstract_Panel
from .mocapproperties import MOCAP_PROPERTIES

_LOG = LogService.get_logger("ui.mocappanel")

# keeps long log lines from unnecesary copying (labels are clipped by panel width anyway)
_MAX_LINE_LENGTH = 120


_TASK_ICONS = {TaskState.PENDING: "DOT", TaskState.RUNNING: "PLAY", TaskState.DONE: "CHECKMARK",
               TaskState.FAILED: "ERROR"}


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    return "%d:%02d" % (seconds // 60, seconds % 60)


def _multiline(layout, lines):
    column = layout.column(align=True)
    column.scale_y = 0.8
    for line in lines:
        column.label(text=line)


class MPFB_PT_Mocap_Panel(Abstract_Panel):
    """Motion capture from video and audio."""
    bl_label = "Motion Capture"
    bl_category = UiService.get_value("MOCAPCATEGORY")
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 85

    def _draw_running(self, layout, job):
        box = self.create_box(layout, "Installing motion capture environment")
        box.label(text="Step %d/%d: %s" % (job.step_index, job.step_count, job.step_label), icon="TIME")
        box.label(text="Elapsed: " + _format_seconds(job.elapsed_seconds))
        if job.last_line:
            box.label(text=job.last_line[:_MAX_LINE_LENGTH])
        if job.step_label.startswith("Installing packages"):
            box.label(text="Downloading packages can take a long time.", icon="INFO")
        row = box.row()
        row.operator("mpfb.mocap_cancel_install", icon="CANCEL")
        row.operator("mpfb.mocap_open_env_dir", text="Open log folder", icon="FILE_FOLDER")

    def _draw_ready(self, layout, env_dir, scene):
        box = self.create_box(layout, "Environment")
        box.label(text="Motion capture environment is ready", icon="CHECKMARK")
        marker = MocapEnvService.read_marker(env_dir) or {}
        versions = marker.get("versions") or {}
        if versions.get("cuda"):
            box.label(text="Torch device: CUDA (%s)" % (versions.get("cuda_device") or "GPU"))
        else:
            box.label(text="Torch device: CPU")
        if versions.get("mediapipe"):
            box.label(text="MediaPipe %s, Torch %s" % (versions.get("mediapipe"), versions.get("torch")))

        self._draw_missing_models(layout)

        run_job = MocapRunService.get_job()
        self._draw_inputs(layout, scene, enabled=not run_job.is_running)
        if run_job.is_running:
            self._draw_run_progress(layout, run_job)
        else:
            row = layout.row()
            row.scale_y = 1.5
            row.operator("mpfb.mocap_run", icon="PLAY")
            self._draw_run_results(layout, run_job)

    def _draw_inputs(self, layout, scene, enabled):
        box = self.create_box(layout, "Video / image")
        box.enabled = enabled
        MOCAP_PROPERTIES.draw_properties(scene, box, ["video_path"], text="")
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["video_pose", "video_hands", "video_face"], toggle=True)
        MOCAP_PROPERTIES.draw_properties(scene, box, ["pose_model", "min_confidence"])
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["video_start", "video_end"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["annotate"])

        box = self.create_box(layout, "Audio")
        box.enabled = enabled
        MOCAP_PROPERTIES.draw_properties(scene, box, ["audio_path"], text="")
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["audio_start", "audio_end"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["audio_device"])

    def _draw_run_progress(self, layout, job):
        box = self.create_box(layout, "Running motion capture")
        box.label(text="Elapsed: " + _format_seconds(job.elapsed_seconds), icon="TIME")
        for task in job.tasks.values():
            if task.status == TaskState.RUNNING:
                if task.total:
                    detail = "%s %d/%d" % (task.stage, task.current, task.total)
                else:
                    detail = task.stage or "starting"
                box.label(text="%s: %s" % (task.label, detail), icon="PLAY")
            else:
                box.label(text="%s: %s" % (task.label, task.status.lower()), icon=_TASK_ICONS.get(task.status, "DOT"))
        if job.last_line:
            box.label(text=job.last_line[:_MAX_LINE_LENGTH])
        row = box.row()
        row.operator("mpfb.mocap_cancel_run", icon="CANCEL")
        row.operator("mpfb.mocap_open_output", text="Open output folder", icon="FILE_FOLDER")

    def _draw_run_results(self, layout, job):
        if job.status == RunJob.IDLE:
            return
        titles = {RunJob.SUCCEEDED: "Last run finished", RunJob.FAILED: "Last run failed",
                  RunJob.CANCELLED: "Last run was cancelled"}
        box = self.create_box(layout, titles.get(job.status, "Last run"))
        box.label(text="Took " + _format_seconds(job.elapsed_seconds))
        for task in job.tasks.values():
            box.label(text="%s: %s" % (task.label, task.status.lower()), icon=_TASK_ICONS.get(task.status, "DOT"))
            if task.message:
                box.label(text=task.message[:_MAX_LINE_LENGTH])
        if job.status == RunJob.FAILED and not job.failed_tasks and job.error:
            box.label(text=job.error[:_MAX_LINE_LENGTH], icon="ERROR")
        if job.loaded_texts:
            box.label(text="Summaries are in the Text Editor:", icon="TEXT")
            for name in job.loaded_texts:
                box.label(text="  " + name)
        if job.loaded_images:
            box.label(text="Plots are in the Image Editor:", icon="IMAGE_DATA")
            for name in job.loaded_images:
                box.label(text="  " + name)
        videos = [output for output in job.outputs if output["kind"] == "video"]
        if videos:
            box.label(text="Annotated video is in the output folder.", icon="FILE_MOVIE")
        box.operator("mpfb.mocap_open_output", text="Open output folder", icon="FILE_FOLDER")

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
            box.label(text=job.error[:_MAX_LINE_LENGTH], icon="ERROR")
            box.label(text="See install.log in the environment folder.")
        box.operator("mpfb.mocap_open_env_dir", text="Open log folder", icon="FILE_FOLDER")

    def _draw_setup(self, layout, job, env_dir):
        self._draw_last_failure(layout, job, env_dir)

        box = self.create_box(layout, "Setup")
        _multiline(box, [
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
            self._draw_ready(layout, env_dir, context.scene)
        else:
            self._draw_setup(layout, job, env_dir)


ClassManager.add_class(MPFB_PT_Mocap_Panel)
