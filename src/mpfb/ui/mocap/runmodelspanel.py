# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         24.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  motion capture subpanel for running the models on video and audio
# ------------------------------------------------------------------------------

import os

from ... import ClassManager
from ...services.logservice import LogService
from ...services.uiservice import UiService
from ...services.mocapenvservice import MocapEnvService
from ...services.mocaprunservice import MocapRunService, RunJob, TaskState
from ...ui.abstractpanel import Abstract_Panel
from .mocapproperties import MOCAP_PROPERTIES, read_settings
from .sourceproperties import VIDEO_INPUTS, face_video_index, absolute
from .mocappanel import format_seconds, MAX_LINE_LENGTH

_LOG = LogService.get_logger("ui.runmodelspanel")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_TASK_ICONS = {TaskState.PENDING: "DOT", TaskState.RUNNING: "PLAY", TaskState.DONE: "CHECKMARK",
               TaskState.FAILED: "ERROR"}


class MPFB_PT_Mocap_Run_Models_Panel(Abstract_Panel):
    """Runs models over the selected video/image/audio."""
    bl_label = "Run Models"
    bl_category = UiService.get_value("MOCAPCATEGORY")
    bl_parent_id = "MPFB_PT_Mocap_Panel"
    bl_options = set()
    bl_order = 1

    @classmethod
    def poll(cls, context):
        """Only once the environment the models run in is installed."""
        return not MocapEnvService.get_job().is_running and MocapEnvService.is_env_installed()

    def _draw_inputs(self, layout, scene, enabled):
        face_index = face_video_index(scene)
        box = self.create_box(layout, "Video / image")
        box.enabled = enabled
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["video_path"], text="")
        row.operator("mpfb.mocap_add_video", text="", icon="ADD")
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["video_pose", "video_hands"], toggle=True)
        face = row.row(align=True)
        # Only one video can be used for the face
        face.enabled = face_index < 0
        MOCAP_PROPERTIES.draw_properties(scene, face, ["video_face"], toggle=True)
        if face_index >= 0:
            box.label(text="Face is taken by video %d of the list." % (face_index + 1), icon="INFO")
        MOCAP_PROPERTIES.draw_properties(scene, box, ["pose_model", "min_confidence"])
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["video_start", "video_end"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["annotate"])

        self._draw_video_list(layout, scene, enabled)

        box = self.create_box(layout, "Audio")
        box.enabled = enabled
        MOCAP_PROPERTIES.draw_properties(scene, box, ["audio_path"], text="")
        row = box.row(align=True)
        MOCAP_PROPERTIES.draw_properties(scene, row, ["audio_start", "audio_end"])
        MOCAP_PROPERTIES.draw_properties(scene, box, ["audio_device"])

    def _draw_video_list(self, layout, scene, enabled):
        """The videos added with + with the models it runs and eventual Face label."""
        videos = getattr(scene, VIDEO_INPUTS)
        if not len(videos):
            return
        box = self.create_box(layout, "Videos to process")
        box.enabled = enabled
        for index, video in enumerate(videos):
            path = absolute(video.path)
            is_image = os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS
            column = box.column(align=True)
            row = column.row(align=True)
            row.label(text="%d. %s" % (index + 1, os.path.basename(path) or "(no file)"),
                      icon="IMAGE_DATA" if is_image else "FILE_MOVIE")
            if video.face:
                row.label(text="Face", icon="USER")
            row.operator("mpfb.mocap_remove_video", text="", icon="X", emboss=False).index = index
            models = [name for name, used in (("pose " + video.pose_model.lower(), video.pose), ("hands", video.hands)) if used]
            details = " · ".join(models) or "no body model"
            if not is_image and (video.start > 0 or video.end > 0):
                details += " · %.1f - %s s" % (video.start, ("%.1f" % video.end) if video.end > 0 else "end")
            line = column.row()
            line.scale_y = 0.7
            line.label(text="     " + details)
            if not os.path.isfile(path):
                column.label(text="The file does not exist", icon="ERROR")

    def _draw_run_summary(self, layout, scene):
        """What Run Mocap will process: every video in the list, the one in the picker, and the audio."""
        settings = read_settings(scene)
        parts = []
        if settings["videos"]:
            parts.append("%d video%s" % (len(settings["videos"]), "s" if len(settings["videos"]) > 1 else ""))
        if settings["audio_path"]:
            parts.append("audio")
        layout.label(text="Will process: " + (" + ".join(parts) if parts else "nothing yet"))

    def _draw_run_progress(self, layout, job):
        box = self.create_box(layout, "Running motion capture")
        box.label(text="Elapsed: " + format_seconds(job.elapsed_seconds), icon="TIME")
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
            box.label(text=job.last_line[:MAX_LINE_LENGTH])
        row = box.row()
        row.operator("mpfb.mocap_cancel_run", icon="CANCEL")
        row.operator("mpfb.mocap_open_output", text="Open output folder", icon="FILE_FOLDER")

    def _draw_run_results(self, layout, job):
        if job.status == RunJob.IDLE:
            return
        titles = {RunJob.SUCCEEDED: "Last run finished", RunJob.FAILED: "Last run failed",
                  RunJob.CANCELLED: "Last run was cancelled"}
        box = self.create_box(layout, titles.get(job.status, "Last run"))
        box.label(text="Took " + format_seconds(job.elapsed_seconds))
        for task in job.tasks.values():
            box.label(text="%s: %s" % (task.label, task.status.lower()), icon=_TASK_ICONS.get(task.status, "DOT"))
            if task.message:
                box.label(text=task.message[:MAX_LINE_LENGTH])
        if job.status == RunJob.FAILED and not job.failed_tasks and job.error:
            box.label(text=job.error[:MAX_LINE_LENGTH], icon="ERROR")
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

    def draw(self, context):
        _LOG.enter()
        layout = self.layout
        run_job = MocapRunService.get_job()
        self._draw_inputs(layout, context.scene, enabled=not run_job.is_running)
        if run_job.is_running:
            self._draw_run_progress(layout, run_job)
        else:
            self._draw_run_summary(layout, context.scene)
            row = layout.row()
            row.scale_y = 1.5
            row.operator("mpfb.mocap_run", icon="PLAY")
            self._draw_run_results(layout, run_job)


ClassManager.add_class(MPFB_PT_Mocap_Run_Models_Panel)
