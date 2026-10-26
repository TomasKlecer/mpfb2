# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operators for running the motion capture worker
# ------------------------------------------------------------------------------

import os
import bpy

from .... import ClassManager
from ....services.logservice import LogService
from ....services.systemservice import SystemService
from ....services.mocapenvservice import MocapEnvService
from ....services.mocaprunservice import MocapRunService, RunJob
from ...mpfboperator import MpfbOperator
from ..mocapproperties import MOCAP_PROPERTIES, read_settings, picked_video
from ..sourceproperties import VIDEO_INPUTS, fill_sources
from ....services.mocapsourceservice import MocapSourceService
from .install_env_operator import _redraw_ui

_LOG = LogService.get_logger("mocap.run_mocap_operator")

_REDRAW_INTERVAL = 0.5


def _fill_animation_sources(job):
    """Put outputs of the finished run into the Create Animation slot."""
    if job.status == RunJob.CANCELLED or bpy.context.scene is None:
        return
    try:
        filled = fill_sources(bpy.context.scene, MocapSourceService.sources_from_run(job.run_dir))
        _LOG.info("Create Animation sources filled from the run", (job.run_dir, filled))
    except Exception as err:  # pylint: disable=W0718
        _LOG.error("Could not fill the Create Animation sources from the run", err)


def _poll_run_job():
    """Timer callback while the worker runs. When its done, the results are loaded into blender."""
    _redraw_ui()
    job = MocapRunService.get_job()
    if job.is_running:
        return _REDRAW_INTERVAL
    MocapRunService.load_results(job)
    _fill_animation_sources(job)
    _redraw_ui()
    _LOG.info("Worker run finished with status", job.status)
    return None


class MPFB_OT_Mocap_Run_Operator(MpfbOperator):
    """Send the selected video / image and audio to the motion capture models. Runs in the background,
    the raw outputs (summary text, plots, annotated video, csv and npz files) are written into new folder in MPFB2 cache"""
    bl_idname = "mpfb.mocap_run"
    bl_label = "Run Mocap"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if MocapEnvService.get_job().is_running:
            cls.poll_message_set("The environment install is running")
            return False
        if MocapRunService.get_job().is_running:
            cls.poll_message_set("Motion capture is already running")
            return False
        if not MocapEnvService.is_env_installed():
            cls.poll_message_set("Install the motion capture environment first")
            return False
        return True

    def hardened_execute(self, context):
        settings = read_settings(context.scene)
        problems = MocapRunService.validate_settings(settings)
        if problems:
            self.report({'ERROR'}, problems[0])
            return {'CANCELLED'}
        missing = MocapEnvService.get_missing_models()
        if missing:
            self.report({'ERROR'}, "Models are missing in data/ai_models: " + ", ".join(missing))
            return {'CANCELLED'}

        run_dir = MocapRunService.new_run_dir()
        job_spec = MocapRunService.build_job(settings, run_dir, MocapEnvService.get_ai_models_dir())
        _LOG.info("Starting the motion capture worker", run_dir)
        MocapRunService.start_run(job_spec)
        if not bpy.app.timers.is_registered(_poll_run_job):
            bpy.app.timers.register(_poll_run_job, first_interval=_REDRAW_INTERVAL)
        self.report({'INFO'}, "Motion capture started, results go to " + run_dir)
        return {'FINISHED'}


class MPFB_OT_Mocap_Cancel_Run_Operator(MpfbOperator):
    """Stop the running motion capture worker"""
    bl_idname = "mpfb.mocap_cancel_run"
    bl_label = "Cancel"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return MocapRunService.get_job().is_running

    def hardened_execute(self, context):
        MocapRunService.cancel_run()
        self.report({'INFO'}, "Cancelling motion capture")
        return {'FINISHED'}


class MPFB_OT_Mocap_Open_Output_Operator(MpfbOperator):
    """Open the folder with the outputs of last motion capture run."""
    bl_idname = "mpfb.mocap_open_output"
    bl_label = "Open output folder"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        job = MocapRunService.get_job()
        return job.status != RunJob.IDLE and os.path.isdir(job.run_dir)

    def hardened_execute(self, context):
        SystemService.open_file_browser(MocapRunService.get_job().run_dir)
        return {'FINISHED'}


class MPFB_OT_Mocap_Add_Video_Operator(MpfbOperator):
    """Add the selected video/image with its current settings to the list of videos to process. Only one video can be used for face."""
    bl_idname = "mpfb.mocap_add_video"
    bl_label = "Add video"
    bl_options = {'REGISTER', 'UNDO'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if not picked_video(context.scene)["path"]:
            cls.poll_message_set("Select a video or an image first")
            return False
        return True

    def hardened_execute(self, context):
        scene = context.scene
        video = picked_video(scene)
        problems = MocapRunService.validate_video(video)
        if problems:
            self.report({'ERROR'}, problems[0])
            return {'CANCELLED'}
        item = getattr(scene, VIDEO_INPUTS).add()
        for key, value in video.items():
            setattr(item, key, value)
        MOCAP_PROPERTIES.set_value("video_path", "", entity_reference=scene)
        if item.face:
            MOCAP_PROPERTIES.set_value("video_face", False, entity_reference=scene)
        return {'FINISHED'}


class MPFB_OT_Mocap_Remove_Video_Operator(MpfbOperator):
    """Remove the video from the list of videos."""
    bl_idname = "mpfb.mocap_remove_video"
    bl_label = "Remove video"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(default=-1)

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        videos = getattr(context.scene, VIDEO_INPUTS)
        if 0 <= self.index < len(videos):
            videos.remove(self.index)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Run_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Add_Video_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Remove_Video_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Cancel_Run_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Open_Output_Operator)
